import argparse
import os
import math
import json
import logging
import random

import torch
import datasets
import accelerate
import wandb

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
wandb.login()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument(
        "--experiment_config",
        type=str,
        default=None,
        help="Path to experiment config file."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default=None
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model checkpoints will be written."
    )
    parser.add_argument(
        "--preprocessed_dataset_path",
        type=str,
        default=None,
        help="Path to preprocessed dataset.")
    parser.add_argument(
        "--save_preprocessed_dataset_path",
        type=str,
        default=None,
        help="Path to save preprocessed dataset."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set."
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--full_training_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for the full training.",)
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum learning rate during training.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--pure_embedding_training_percentage",
        type=int,
        default=0,
        help="Percentage of training steps to train only the embedding layer."
    )
    parser.add_argument(
        "--warmup_percentage",
        type=int,
        default=10,
        help="Percentage of training steps to warmup to the learning rate."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
    )
    parser.add_argument(
        "--eval_steps",
        type=str,
        default=None,
        help="Whether to evaluate the model at the end of every n steps, or 'epoch' for each epoch."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    args = parser.parse_args()

    # If provided, load experiment settings from config file
    if args.experiment_config is not None:
        with open(args.experiment_config) as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None and args.preprocessed_dataset_path is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    return args


def save_checkpoint(output_dir, model, optimizer, lr_scheduler, epoch, loss, completed_steps=None):
    if completed_steps is not None:
        output_file = f"step_{completed_steps}.pt"
    else:
        output_file = f"epoch_{epoch}.pt"
    if output_dir is not None:
        output_file = os.path.join(output_dir, output_file)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
    }, output_file)


def validate_model(model, eval_dataloader, device):
    model.eval()
    losses = []
    num_eval_steps = len(eval_dataloader)
    eval_progress_bar = tqdm(range(num_eval_steps), position=0, leave=True)
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss)
        eval_progress_bar.update(1)

    losses = torch.stack(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return eval_loss, perplexity

def run_clm(args):
    wandb.init(
        project="CLP study project",
        config=vars(args),
    )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory if needed
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.block_size is None:
        args.block_size = tokenizer.model_max_length

    if args.preprocessed_dataset_path is not None:
        lm_dataset = datasets.load_from_disk(args.preprocessed_dataset_path)
        logger.info(f"Loaded preprocessed dataset from {args.preprocessed_dataset_path}")
    else:
        if args.dataset_name:
            dataset = datasets.load_dataset(
                args.dataset_name,
                args.dataset_config_name
            )
            if "validation" not in dataset.keys():
                dataset["validation"] = datasets.load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                dataset["train"] = datasets.load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
            logger.info(f"Loaded dataset {args.dataset_name} with config {args.dataset_config_name}")
        else:
            data_files = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            extension = (
                args.train_file.split(".")[-1]
                if args.train_file is not None
                else args.validation_file.split(".")[-1]
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
                    split=f"train[:{args.validation_split_percentage}%]"
                )
                dataset["train"] = datasets.load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{args.validation_split_percentage}%:]"
                )

        def preprocess_function(examples):
            return tokenizer([x for x in examples["text"]])

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= args.block_size:
                total_length = (total_length // args.block_size) * args.block_size
            # Split by chunks of block_size.
            result = {
                k: [t[i: i + args.block_size] for i in range(0, total_length, args.block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=args.preprocessing_num_workers)
        if args.save_preprocessed_dataset_path is not None:
            lm_dataset.save_to_disk(args.save_preprocessed_dataset_path)

    lm_dataset.set_format("torch")
    if args.max_train_samples is not None:
        max_train_samples = min(len(lm_dataset["train"]), args.max_train_samples)
        lm_dataset["train"] = lm_dataset["train"].select(range(max_train_samples))
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(lm_dataset["validation"]), args.max_eval_samples)
        lm_dataset["validation"] = lm_dataset["validation"].select(range(max_eval_samples))


    # Log a few random samples from the training set:
    for index in random.sample(range(len(lm_dataset["train"])), 3):
        logger.info(f"Sample {index} of the training set: {lm_dataset['train'][index]}.")

    tokenizer.pad_token = tokenizer.eos_token   # TODO not in run_clm_no_trainer.py, but in run_clm.py with HF Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        lm_dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        lm_dataset["validation"], collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Load pretrained model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    logger.info(f"Loaded model {args.model_name_or_path}")

    if args.pure_embedding_training_percentage > 0:
        # freeze transformer layer to only train embeddings, the language modeling head (embed_out) is not affected
        for param in model.gpt_neox.parameters():
            param.requires_grad = False
        # unfreeze word embeddings
        model.gpt_neox.embed_in.weight.requires_grad = True
        transformer_layers_are_frozen = True
    else:
        transformer_layers_are_frozen = False

    model.to(args.device)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2))

    # LR Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    pure_embedding_training_steps = math.ceil(num_training_steps * args.pure_embedding_training_percentage / 100)
    warmup_embedding_steps = math.ceil(pure_embedding_training_steps * args.warmup_percentage / 100)
    warmup_and_embedding_training_steps = warmup_embedding_steps + pure_embedding_training_steps
    # custom lr scheduler with cosine decay and one warmup phase each for the embedding training and the full training
    lr_scheduler = get_custom_lr_scheduler(
        optimizer=optimizer,
        warmup_percentage=args.warmup_percentage,
        num_training_steps=num_training_steps,
        pure_embedding_training_percentage=args.pure_embedding_training_percentage,
        min_lr=args.min_lr
    )

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    eval_steps = args.eval_steps
    if eval_steps is not None and eval_steps.isdigit():
        eval_steps = int(eval_steps)

    # Training
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(lm_dataset['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    progress_bar = tqdm(range(num_training_steps), position=0, leave=True)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(os.path.basename(args.resume_from_checkpoint))[0]

        if "epoch" in training_difference:
            starting_epoch = epoch + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerate.data_loader.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        accumulated_loss = 0
        for step, batch in enumerate(active_dataloader):
            if transformer_layers_are_frozen and completed_steps >= warmup_and_embedding_training_steps:
                # unfreeze transformer layers
                for param in model.gpt_neox.parameters():
                    param.requires_grad = True
                # set new learning rate for full training and decay to 0.0 for the full training
                lr_scheduler.base_lrs = [args.full_training_learning_rate for _ in lr_scheduler.base_lrs]
                transformer_layers_are_frozen = False
                logger.info(f"epoch {epoch}: step {completed_steps}: unfreezing transformer layers")

            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # loss = loss_function(outputs, targets)
            loss = loss / args.gradient_accumulation_steps
            accumulated_loss += loss.detach().float()
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            loss.backward()
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(train_dataloader)):
                perplexity = math.exp(accumulated_loss)
                wandb.log({
                    "train/train_loss": accumulated_loss,
                    "train/perplexity": perplexity,
                    "train/lr": optimizer.param_groups[0]["lr"],
                })
                accumulated_loss = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if isinstance(eval_steps, int):
                    if completed_steps % eval_steps == 0 and completed_steps > 0:
                        logger.info(f"epoch {epoch}: step {completed_steps}: evaluating model")
                        eval_loss, perplexity = validate_model(model, eval_dataloader, model.device)
                        logger.info(f"epoch {epoch}: step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss}")
                        wandb.log({
                            "val/eval_loss": eval_loss,
                            "val/perplexity": perplexity,
                        })
                        model.train()

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0 and completed_steps > 0:
                        logger.info(f"epoch {epoch}: step {completed_steps}: saving checkpoint")
                        save_checkpoint(
                            args.output_dir, model, optimizer, lr_scheduler, epoch, loss, completed_steps=completed_steps
                        )

        logger.info(f"epoch {epoch}: step {completed_steps}: evaluating model")
        eval_loss, perplexity = validate_model(model, eval_dataloader, model.device)
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
        wandb.log({
            "val/eval_loss": eval_loss,
            "val/perplexity": perplexity,
        })

        if args.checkpointing_steps == "epoch":
            logger.info(f"epoch {epoch}: step {completed_steps}: saving checkpoint")
            save_checkpoint(args.output_dir, model, optimizer, lr_scheduler, epoch, loss)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump({"perplexity": perplexity}, f)

    wandb.finish()


def main():
    args = parse_args()
    run_clm(args)


if __name__ == '__main__':
    main()
