# Baseline CLP (0% embedding tuning, only full model training)
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/wikitext_de_random_init.json \
  --output_dir=../outputs/wikitext_de_baseline_random_init \
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
  --experiment_config=../configs/wikitext_de_random_init.json \
  --output_dir=../outputs/wikitext_de_embedding_tuning_random_init \
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

# Baseline CLP (0% embedding tuning, only full model training)
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/oscar_de_random_init.json \
  --output_dir=../outputs/oscar_de_baseline_random_init \
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
  --experiment_config=../configs/oscar_de_random_init.json \
  --output_dir=../outputs/oscar_de_embedding_tuning_random_init \
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