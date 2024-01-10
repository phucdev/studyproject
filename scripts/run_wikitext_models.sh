# Baseline CLP (0% embedding tuning, only full model training)
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_baseline \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=0 \
  --embedding_tuning_percentage=0 \
  --learning_rate=3e-4 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup" \
  --calculate_warmup_based_on_num_train_steps

# Embedding Tuning (only train the embedding layer at the start with a high learning rate)
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Accelerated Training Start (no freezing/ embedding tuning, only high learning rate at the start)
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_accelerated_training_start \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --not_freeze_transformer_layers \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Frozen Transformer Layers (100% embedding tuning)
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_frozen_transformer_layers \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=100 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Different Warmup Strategies for Embedding Tuning
# Only full training warmup
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning_full_training_warmup \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=0 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Only embedding tuning warmup
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning_et_warmup \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=0 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# No warmup
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning_no_warmup \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=0 \
  --embedding_tuning_warmup_percentage=0 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Different Learning Rate Schedules
# Constant schedule with warmup
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning_constant \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=0.001 \
  --min_lr=0.001 \
  --lr_scheduler_type="constant_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Mixed: constant schedule with warmup for embedding tuning, cosine schedule with warmup for full training
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning_mixed \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=0.001 \
  --lr_scheduler_type="mixed_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Custom embedding tuning denominator: slower decay of learning rate for embedding tuning
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning_custom_et_denominator \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --set_embedding_tuning_denominator \
  --calculate_warmup_based_on_num_train_steps

# Unified learning rate schedule for the whole training
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning_unified \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=0.001 \
  --min_lr=3e-5 \
  --lr_scheduler_type="unified_cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Different Embedding Tuning Percentages
# 25% embedding tuning
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_25_embedding_tuning \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=25 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# 50% embedding tuning
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_50_embedding_tuning \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=50 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# 75% embedding tuning
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_experiments.json \
  --output_dir=../outputs/wikitext_de_75_embedding_tuning \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=75 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Variable batch size training (1 for embedding tuning, 64 for full model training)
# embedding tuning with normal lr scheduler
accelerate launch ../run_variable_batch_size_lm.py \
  --experiment_config=../configs/wikitext_de_embedding_tuning_variable_batch_size.json \
  --output_dir=../outputs/wikitext_de_variable_batch_size \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=0.001 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --per_device_embedding_tuning_batch_size=1

# embedding tuning with custom embedding tuning denominator
accelerate launch ../run_variable_batch_size_lm.py \
  --experiment_config=../configs/wikitext_de_embedding_tuning_variable_batch_size.json \
  --output_dir=../outputs/wikitext_de_variable_batch_size_et_denominator \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=0.001 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --set_embedding_tuning_denominator \
  --per_device_embedding_tuning_batch_size=1

# unified lr scheduler
accelerate launch ../run_variable_batch_size_lm.py \
  --experiment_config=../configs/wikitext_de_embedding_tuning_variable_batch_size.json \
  --output_dir=../outputs/wikitext_de_variable_batch_size_unified \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=0.001 \
  --min_lr=0.001 \
  --lr_scheduler_type="unified_cosine_schedule_with_warmup_embedding_tuning" \
  --per_device_embedding_tuning_batch_size=1

# no freezing of transformer layers
accelerate launch ../run_variable_batch_size_lm.py \
  --experiment_config=../configs/wikitext_de_embedding_tuning_variable_batch_size.json \
  --output_dir=../outputs/wikitext_de_variable_batch_size_accelerated_training_start \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=0.001 \
  --not_freeze_transformer_layers \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --per_device_embedding_tuning_batch_size=1

# no freezing of transformer layers, custom embedding tuning denominator
accelerate launch ../run_variable_batch_size_lm.py \
  --experiment_config=../configs/wikitext_de_embedding_tuning_variable_batch_size.json \
  --output_dir=../outputs/wikitext_de_variable_batch_size_accelerated_training_start_et_denominator \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=0.001 \
  --not_freeze_transformer_layers \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --set_embedding_tuning_denominator \
  --per_device_embedding_tuning_batch_size=1