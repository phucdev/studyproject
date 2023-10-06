import datasets
import torch
import math
import fire
import logging
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    set_seed,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

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

    block_size = 128

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

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load pretrained model and prepare the trainer
    set_seed(seed)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)

    # freeze transformer layer to only train embeddings, the language modeling head (embed_out) is not affected
    for param in model.gpt_neox.parameters():
        param.requires_grad = False
    # unfreeze word embeddings
    model.gpt_neox.embed_in.weight.requires_grad = True

    # TODO adjust LR scheduler and add training with unfrozen transformer layer
    #  if that kind of behavior is not possible with the Trainer API we will have implement this with pytorch
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
