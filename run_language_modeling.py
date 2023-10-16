import sys
import os
import math
import json
import logging

import fire
import datasets
import torch

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    set_seed,
    DataCollatorForLanguageModeling
)
from custom_lr_scheduler import get_custom_lr_scheduler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_clm(
    model_name_or_path,
    output_dir,
    train_file=None,
    validation_file=None,
    dataset_name=None,
    dataset_config_name=None,
    validation_split_percentage=10,
    max_train_samples=None,
    max_eval_samples=None,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    full_training_learning_rate=1e-4,
    min_lr=0.0,
    weight_decay=0.1,
    gradient_accumulation_steps = 1,
    pure_embedding_training_percentage=10,
    warmup_percentage=10,
    num_train_epochs=1,
    block_size=None,
    device="cuda",
    seed=42
):
    # Load dataset
    if dataset_name:
        dataset = datasets.load_dataset(
            dataset_name,
            dataset_config_name
        )
        if "validation" not in dataset.keys():
            dataset["validation"] = datasets.load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{validation_split_percentage}%]",
            )
            dataset["train"] = datasets.load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
        if validation_file is not None:
            data_files["validation"] = validation_file
        extension = (
            train_file.split(".")[-1]
            if train_file is not None
            else validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
        dataset = datasets.load_dataset(
            extension,
            data_files=data_files
        )
        if "validation" not in dataset.keys():
            dataset["validation"] = datasets.load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{validation_split_percentage}%]"
            )
            dataset["train"] = datasets.load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{validation_split_percentage}%:]"
            )

    # Preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess_function(examples):
        return tokenizer([x for x in examples["text"]])

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
    )

    if block_size is None:
        block_size = tokenizer.model_max_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=1)
    lm_dataset.set_format("torch")
    if max_train_samples is not None:
        max_train_samples = min(len(lm_dataset["train"]), max_train_samples)
        lm_dataset["train"] = lm_dataset["train"].select(range(max_train_samples))
    if max_eval_samples is not None:
        max_eval_samples = min(len(lm_dataset["validation"]), max_eval_samples)
        lm_dataset["validation"] = lm_dataset["validation"].select(range(max_eval_samples))

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        lm_dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        lm_dataset["validation"], collate_fn=data_collator, batch_size=per_device_eval_batch_size
    )

    # Load pretrained model
    set_seed(seed)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)

    if pure_embedding_training_percentage > 0:
        # freeze transformer layer to only train embeddings, the language modeling head (embed_out) is not affected
        for param in model.gpt_neox.parameters():
            param.requires_grad = False
        # unfreeze word embeddings
        model.gpt_neox.embed_in.weight.requires_grad = True
    transformer_layers_are_frozen = True

    model.to(device)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # LR Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    pure_embedding_training_steps = math.ceil(num_training_steps * pure_embedding_training_percentage / 100)
    warmup_embedding_steps = math.ceil(pure_embedding_training_steps * warmup_percentage / 100)
    warmup_and_embedding_training_steps = warmup_embedding_steps + pure_embedding_training_steps
    # custom lr scheduler with cosine decay and one warmup phase each for the embedding training and the full training
    lr_scheduler = get_custom_lr_scheduler(
        optimizer=optimizer,
        warmup_percentage=warmup_percentage,
        num_training_steps=num_training_steps,
        num_pure_embedding_training_steps=pure_embedding_training_steps,
        min_lr=min_lr
    )

    # Training
    total_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(lm_dataset['train'])}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    progress_bar = tqdm(range(num_training_steps))
    completed_steps = 0

    for epoch in range(num_train_epochs):
        model.train()
        total_loss = 0
        step = 0
        for step, batch in enumerate(train_dataloader):
            if transformer_layers_are_frozen and completed_steps >= warmup_and_embedding_training_steps:
                # unfreeze transformer layers
                for param in model.gpt_neox.parameters():
                    param.requires_grad = True
                # set new learning rate for full training and decay to 0.0 for the full training
                lr_scheduler.base_lrs = [full_training_learning_rate for _ in lr_scheduler.base_lrs]
                lr_scheduler.min_lr = 0.0   # TODO check if we want to decay to 0 for the full training
                transformer_layers_are_frozen = False
                logger.info(f"epoch {epoch}: step {completed_steps}: unfreezing transformer layers")

            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # loss = loss_function(outputs, targets)
            loss = loss / gradient_accumulation_steps
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                before_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                lr_scheduler.step()
                after_lr = optimizer.param_groups[0]["lr"]
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                logger.info(f"epoch {epoch}: step {completed_steps}: lr {before_lr} -> {after_lr} loss {loss}")

        # gradient accumulation for the last batch
        if (step + 1) % gradient_accumulation_steps != 0:
            before_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            lr_scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1
            logger.info(f"epoch {epoch}: step {completed_steps}: lr {before_lr} -> {after_lr} loss {loss}")

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss)

        losses = torch.stack(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity}, f)



if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
