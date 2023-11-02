# Efficient Cross-Lingual and Progressive Transfer Learning

This study project is based on the CLP-Transfer method by Ostendorff and Rehm (2023), that transfers models from a source language, for which pretrained models are publicly available, like English, to a new target language while leveraging a small pretrained model in the target language as a helper model. 

- **Preprint:** https://arxiv.org/abs/2301.09626

```bibtex
@misc{Ostendorff2023clp,
  doi = {10.48550/ARXIV.2301.09626},
  author = {Ostendorff, Malte and Rehm, Georg},
  title = {Efficient Language Model Training through Cross-Lingual and Progressive Transfer Learning},
  publisher = {arXiv},
  year = {2023}
}
```

We aim to improve upon the CLP-Transfer method through several modifications. 


## Usage

### Pretraining dataset

We follow the methodology from Minixhofer et al. (2021) and Ostendorff & Rehm (2023) to construct a separate training and validation dataset from the German subset of OSCAR v2019 (Ortiz Su'arez et al., 2019). They use the first 4GB of OSCAR as the training dataset and the next 0.4GB as the validation set. The training dataset contains approximately 30.8B tokens. To reproduce the pretraining dataset for the German model:

````bash
python prepare_oscar.py \
	--dataset_name=oscar \
	--dataset_config_name=unshuffled_deduplicated_de \
	--output_dir=data/oscar_de \
	--subsample_size_mb=4096
````

### CLP

To apply CLP-Transfer, you need a large source model (e.g., in English) and a small model in your target language.

```bash
# helper: other model in target language but with same tokenizer (smaller or other architecture)
# source: same size as target model but different language/multilingual
python clp.py apply_clp \
    --source_model_name_or_path EleutherAI/pythia-410m \
    --helper_model_name_or_path malteos/gpt2-wechsel-german-ds-me \
    --target_model_path <output_dir>
```

### Pretraining

To train a model on the causal language modeling objective, run:

```bash
python run_language_modeling.py --experiment_config=configs/oscar_de_baseline.json
```

If you want to train a model on the causal language modeling objective, but with a pure embedding training phase, where we 
first freeze the transformer layers and train the word embeddings with a high learning rate for a percentage of the 
number of training steps and then train the full model with a lower learning rate, then run:

```bash
python run_language_modeling.py --experiment_config=configs/oscar_de_embedding_tuning.json
```

You can pass arguments directly via the CLI or by specifying a JSON config file with the arguments.
If you want to track the experiment, set the `--with_tracking` flag and set the `--report_to` parameter
to the platform of your choice.

In order to train in a distributed setup with `accelerate`, create a config with
```bash
accelerate config
```
Then you can launch the script with `accelerate launch`. A complete example would be:
```bash
accelerate launch run_language_modeling.py --experiment_config=configs/oscar_de_embedding_tuning.json --with_tracking --report_to=wandb
```


### Evaluation

#### Perplexity



````bash
TODO
````



#### Performance on downstream tasks

https://github.com/OpenGPTX/lm-evaluation-harness

````bash
TODO
````



## References



## License

Code: MIT
