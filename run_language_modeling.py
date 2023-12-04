import argparse
import os
import math
import json
import logging
import random

import torch
import transformers
import datasets

from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.scheduler import AcceleratedScheduler
from huggingface_hub import Repository, create_repo
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    set_seed,
    DataCollatorForLanguageModeling
)
from custom_lr_scheduler import get_custom_lr_scheduler

logger = get_logger(__name__)


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
        "--embedding_tuning_percentage",
        type=int,
        default=0,
        help="Percentage of training steps to train only the embedding layer."
    )
    parser.add_argument(
        "--not_freeze_transformer_layers",
        action="store_true",
        default=False,
        help="Do not freeze transformer layers during pure embedding training phase (for testing purposes)."
    )
    parser.add_argument(
        "--embedding_tuning_warmup_percentage",
        type=int,
        default=10,
        help="Percentage of embedding tuning steps to warmup to the learning rate."
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
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
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
        "--eval_iters",
        type=int,
        default=None,
        help="Number of evaluation iterations."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="CLP study project",
        help="The name of the project to log to. Only applicable when `--with_tracking` is passed."
    )
    args = parser.parse_args()

    # If provided, load experiment settings from config file.
    # Be aware that the parameters in the config overwrite the default and CLI parameters.
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

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def validate_model(model, eval_dataloader, accelerator, per_device_eval_batch_size, eval_iters=None):
    """Evaluates the model on eval_dataloader and returns the evaluation loss and perplexity.
    Args:
        model (:obj:`torch.nn.Module`): The model to evaluate.
        eval_dataloader (:obj:`torch.utils.data.DataLoader`): The evaluation dataloader.
        accelerator (:obj:`accelerate.Accelerator`): The distributed training backend.
        per_device_eval_batch_size (:obj:`int`): The batch size per device.
        eval_iters (:obj:`int`, `optional`): The number of iterations to run evaluation for. Defaults to `None` which
            means that the whole dataset will be used.
    Returns:
        :obj:`tuple(torch.FloatTensor, float)`: A tuple with the evaluation loss and the perplexity.
    """
    model.eval()
    losses = []
    num_eval_steps = len(eval_dataloader)
    if eval_iters is not None:
        num_eval_steps = min(num_eval_steps, eval_iters)
    eval_progress_bar = tqdm(range(num_eval_steps), disable=not accelerator.is_local_main_process,
                             position=0, leave=True, desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        if eval_iters is not None and step >= eval_iters:
            break

        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))
        eval_progress_bar.update(1)

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return eval_loss, perplexity


def run_clm(args):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed to {args.seed}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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
            logger.info(f"Loaded dataset from files {data_files}")

        def preprocess_function(examples):
            return tokenizer([x for x in examples["text"]])

        with accelerator.main_process_first():
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=dataset["train"].column_names,
                desc="Running tokenizer on dataset",
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

        with accelerator.main_process_first():
            lm_dataset = tokenized_dataset.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                desc=f"Grouping texts in chunks of {args.block_size}",
            )
            if args.save_preprocessed_dataset_path is not None:
                lm_dataset.save_to_disk(args.save_preprocessed_dataset_path)
                logger.info(f"Saved preprocessed dataset to {args.save_preprocessed_dataset_path}")

    with accelerator.main_process_first():
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
    # we shuffle to get a new random sample each time we evaluate for eval_iters
    eval_dataloader = DataLoader(
        lm_dataset["validation"], shuffle=True, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Load pretrained model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    logger.info(f"Loaded model {args.model_name_or_path}")

    if args.embedding_tuning_percentage > 0:
        if args.not_freeze_transformer_layers:  # for testing purposes
            logger.info(f"Accelerated training for {args.embedding_tuning_percentage}% of training steps without freezing transformer layers")
        else:
            # freeze transformer layer to only train embeddings, the language modeling head (embed_out) is not affected
            for param in model.gpt_neox.parameters():
                param.requires_grad = False
            # unfreeze word embeddings
            model.gpt_neox.embed_in.weight.requires_grad = True
            logger.info(f"Freezing transformer layers for {args.embedding_tuning_percentage}% of training steps")
        transformer_layers_are_frozen = True
    else:
        transformer_layers_are_frozen = False

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
    # custom lr scheduler with cosine decay and one warmup phase each for the embedding training and the full training
    lr_scheduler = get_custom_lr_scheduler(
        optimizer=optimizer,
        warmup_percentage=args.warmup_percentage,
        num_training_steps=num_training_steps,
        embedding_tuning_percentage=args.embedding_tuning_percentage,
        min_lr=args.min_lr,
        embedding_tuning_warmup_percentage=args.embedding_tuning_warmup_percentage
    )
    logger.info(f"num_training_steps: {num_training_steps} before accelerator.prepare")

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    embedding_tuning_steps = math.ceil(num_training_steps * args.embedding_tuning_percentage / 100)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(num_training_steps / num_update_steps_per_epoch)
    logger.info(f"num_training_steps: {num_training_steps} after accelerator.prepare")

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    eval_steps = args.eval_steps
    if eval_steps is not None and eval_steps.isdigit():
        eval_steps = int(eval_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = "2 phase custom LR scheduler with warmup, cosine decay"
        accelerator.init_trackers(args.project_name, experiment_config)

    # Training
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(lm_dataset['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process,
                        position=0, leave=True, desc="Training")
    completed_steps = 0
    starting_epoch = 0
    consumed_train_tokens = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(args.resume_from_checkpoint)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
        consumed_train_tokens += completed_steps * total_batch_size * args.block_size

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        batch_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            if transformer_layers_are_frozen and completed_steps >= embedding_tuning_steps:
                accelerator.wait_for_everyone()
                # unfreeze transformer layers
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    for param in model.module.gpt_neox.parameters():
                        param.requires_grad = True
                else:
                    for param in model.gpt_neox.parameters():
                        param.requires_grad = True
                # set new peak learning rate for full training
                if isinstance(lr_scheduler, AcceleratedScheduler):
                    lr_scheduler.scheduler.base_lrs = [args.full_training_learning_rate for _ in
                                                       lr_scheduler.scheduler.base_lrs]
                else:
                    lr_scheduler.base_lrs = [args.full_training_learning_rate for _ in lr_scheduler.base_lrs]
                transformer_layers_are_frozen = False
                if args.not_freeze_transformer_layers:
                    logger.info(f"epoch {epoch}: step {completed_steps}: finish accelerated training")
                else:
                    logger.info(f"epoch {epoch}: step {completed_steps}: unfreezing transformer layers")
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                train_step_loss = loss.detach().float()
                batch_loss += train_step_loss / accelerator.gradient_accumulation_steps
                # We keep track of the loss at each epoch
                total_loss += train_step_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.with_tracking and accelerator.is_main_process:
                    # https://github.com/huggingface/accelerate/issues/639 loss tracking with gradient accumulation
                    # script seems to get stuck when calling accelerator.gather on batch loss, so we only track
                    # average train loss on the main process
                    train_loss = float(batch_loss)
                    train_perplexity = math.exp(train_loss)
                    log_dict = {
                        "train/loss": train_loss,
                        "train/perplexity": train_perplexity,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "consumed_train_tokens": completed_steps * total_batch_size * args.block_size   # TODO this might not be accurate
                    }
                    progress_bar.set_postfix(log_dict)
                    accelerator.log(log_dict, step=completed_steps)
                    batch_loss = 0

                if isinstance(eval_steps, int):
                    if completed_steps % eval_steps == 0 and completed_steps > 0:
                        logger.info(f"epoch {epoch}: step {completed_steps}: evaluating model")
                        eval_loss, perplexity = validate_model(
                            model, eval_dataloader, accelerator, args.per_device_eval_batch_size, args.eval_iters
                        )
                        logger.info(
                            f"epoch {epoch}: step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss}")
                        if args.with_tracking:
                            accelerator.log({
                                "eval/loss": eval_loss,
                                "eval/perplexity": perplexity,
                                "epoch": epoch,
                                "step": completed_steps,
                                "consumed_train_tokens":  completed_steps * total_batch_size * args.block_size
                            }, step=completed_steps)
                        model.train()

                if isinstance(checkpointing_steps, int):
                    if completed_steps > 0 and completed_steps % checkpointing_steps == 0:
                        logger.info(f"epoch {epoch}: step {completed_steps}: saving checkpoint")
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

        if args.eval_steps == "epoch":
            # Evaluate on the whole validation split at the end of each epoch
            logger.info(f"epoch {epoch}: step {completed_steps}: evaluating model")
            eval_loss, perplexity = validate_model(
                model, eval_dataloader, accelerator, args.per_device_eval_batch_size
            )
            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "eval/perplexity": perplexity,
                        "eval/loss": eval_loss,
                        "train/epoch_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                        "consumed_train_tokens": completed_steps * total_batch_size * args.block_size
                    },
                    step=completed_steps,
                )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f)


def main():
    args = parse_args()
    run_clm(args)


if __name__ == '__main__':
    main()
