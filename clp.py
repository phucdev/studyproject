"""
Adapted from https://github.com/malteos/clp-transfer/blob/main/clp.py
# Efficient Language Model Training through Cross-Lingual and Progressive Transfer Learning

- overlapping token in both source vocab + target vocab
- trained (small) model with target vocab: retrieve NNs for missing tokens
- build target embeddings for missing tokens through weighted combinations of NNs of source embeddings

"""
import sys
import logging
import os
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

import fire
from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def xavier_normal(tensor):
    """Fills the input Tensor with values according to the method described in Understanding the difficulty of
    training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py#L122"""
    return torch.nn.init.xavier_normal_(tensor)


def small_init(tensor, dim):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution.
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py#L138"""
    # dim is hidden size: in our case it is 1024 for pythia-410m
    std = math.sqrt(2 / (5 * dim))
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)


def apply_clp(
        source_model_name_or_path, 
        helper_model_name_or_path,  
        target_model_path, 
        helper_tokenizer_name_or_path = None,
        seed=42,
        override: bool = False,
        random_init: bool = False,
        random_init_method: str = None
    ):
    """

    Example:
    
    python clp.py apply_clp \
        --source_model_name_or_path ${DATASETS_DIR}/huggingface_transformers/pytorch/bloom-350m \
        --helper_model_name_or_path ${DATASETS_DIR}/huggingface_transformers/pytorch/bloom-350m-clp-german \
        --target_model_path ./data/bloom-350m-clp-oracle-german

    python cli_clp.py apply_clp \
        --source_model_name_or_path ${DATASETS_DIR}/huggingface_transformers/pytorch/bloom-350m \
        --helper_model_name_or_path ${DATASETS_DIR}/huggingface_transformers/pytorch/gpt2-xl-wechsel-german \
        --target_model_path ./data/bloom-350m-clp-german

    helper: other model in target language but with same tokenizer (smaller or other architecture)
    source: target model size but different language
    target: output model
    
    """
    if os.path.exists(target_model_path) and not override:
        raise FileExistsError(f'Output exists already at {target_model_path} fix with --override')

    logger.info(f'Loading source model: {source_model_name_or_path}')

    if "bert" in source_model_name_or_path:
        source_model = AutoModelForMaskedLM.from_pretrained(source_model_name_or_path)
    else:
        source_model = AutoModelForCausalLM.from_pretrained(source_model_name_or_path)
    source_tokenizer = AutoTokenizer.from_pretrained(source_model_name_or_path)
    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()


    if not helper_tokenizer_name_or_path:
        helper_tokenizer_name_or_path = helper_model_name_or_path

    logger.info(f'Loading helper model: {helper_model_name_or_path}')
    logger.info(f'Loading helper tokenizer: {helper_tokenizer_name_or_path}')

    if "bert" in helper_model_name_or_path:
        helper_model = AutoModelForMaskedLM.from_pretrained(helper_model_name_or_path)
    else:
        helper_model = AutoModelForCausalLM.from_pretrained(helper_model_name_or_path)
    helper_tokenizer = AutoTokenizer.from_pretrained(helper_tokenizer_name_or_path)
    helper_embeddings = helper_model.get_input_embeddings().weight.detach().numpy()

    # Overlapping tokens
    target_tokens = set(helper_tokenizer.get_vocab().keys())
    source_tokens = set(source_tokenizer.get_vocab().keys())

    overlapping_tokens = target_tokens & source_tokens
    missing_tokens = target_tokens - source_tokens

    missing_tokens_list = list(missing_tokens)
    overlapping_tokens_list = list(overlapping_tokens)

    logger.info(f'{len(overlapping_tokens)=}; {len(missing_tokens)=}')

    if not overlapping_tokens:
        raise ValueError('No overlapping tokens found')

    source_token_to_idx = {t: i for t, i in source_tokenizer.get_vocab().items()}
    # target_token_to_idx = {t: i for t, i in target_tokenizer.get_vocab().items()}
    helper_token_to_idx = {t: i for t, i in helper_tokenizer.get_vocab().items()}

    overlapping_tokens_idxs = [source_token_to_idx[t] for t in overlapping_tokens_list]
    overlapping_token_vecs = source_embeddings[overlapping_tokens_idxs, :]
    
    logger.info(f'{overlapping_token_vecs.shape=}')

    # Target embeddings
    np.random.seed(seed)
    if random_init:
        logger.info(f'Use randomly initialized target embeddings')
        # Random init target embeddings with mean+std of source embeddings
        if random_init_method == 'xavier':
            target_embeddings = xavier_normal(torch.empty(len(target_tokens), source_embeddings.shape[1])).numpy()
        elif random_init_method == 'small_init':
            target_embeddings = small_init(torch.empty(len(target_tokens), source_embeddings.shape[1]), source_embeddings.shape[1]).numpy()
        else:
            target_embeddings = np.random.normal(size=(len(target_tokens), source_embeddings.shape[1]))
    else:
        # Random init target embeddings with mean+std of source embeddings
        target_embeddings = np.random.normal(
            np.mean(source_embeddings, axis=0),
            np.std(source_embeddings, axis=0),
            (
                len(target_tokens),
                source_embeddings.shape[1]
            )
        )
        # Set overlapping tokens
        for t in overlapping_tokens:
            target_embeddings[helper_token_to_idx[t]] = source_embeddings[source_token_to_idx[t]]

        # TODO possible modifications:
        #  use method from https://github.com/wietsedv/gpt2-recycle to transform smaller token embeddings to larger ones
        #  normalizations?
        if missing_tokens:

            helper_missing_tokens_vecs = helper_embeddings[[helper_token_to_idx[t] for t in missing_tokens_list], :]
            helper_overlapping_token_vecs = helper_embeddings[[helper_token_to_idx[t] for t in overlapping_tokens_list], :]

            # Similarities for missing tokens
            sims = cosine_similarity(helper_missing_tokens_vecs, helper_overlapping_token_vecs)

            # similar = 1 => high weight
            # dissimilar = 0 => low weight

            for ti, t in enumerate(tqdm(missing_tokens_list)):  # 1:14hrs (12min with batch sim)
                # distances to overlapping tokens
                token_sims = sims[ti]
                norm_sims = token_sims / token_sims.sum()

                # weighted average of overlapping token embeddings with weight from similarity in helper token embedding space
                target_vec = np.average(overlapping_token_vecs, axis=0, weights=norm_sims)
                target_embeddings[helper_token_to_idx[t]] = target_vec
        else:
            logger.warning('No missing tokens')

    # Save target model
    target_model = source_model
    target_tokenizer = helper_tokenizer
    target_model.resize_token_embeddings(len(target_tokenizer))
    target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)

    target_model.save_pretrained(target_model_path)
    target_tokenizer.save_pretrained(target_model_path)
    logger.info(f'Saved to {target_model_path}')


if __name__ == '__main__':
    fire.Fire(apply_clp)
    sys.exit(0)
