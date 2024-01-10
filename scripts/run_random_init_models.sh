# Baseline CLP (0% embedding tuning, only full model training)
accelerate launch ./run_language_modeling.py \
  --experiment_config=./configs/wikitext_de_random_init.json \
  --output_dir=./outputs/wikitext_de_baseline_random_init \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=0 \
  --embedding_tuning_percentage=0 \
  --learning_rate=3e-4 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup" \
  --calculate_warmup_based_on_num_train_steps

# Embedding Tuning (only train the embedding layer at the start with a high learning rate)
accelerate launch ./run_language_modeling.py \
  --experiment_config=./configs/wikitext_de_random_init.json \
  --output_dir=./outputs/wikitext_de_embedding_tuning_random_init \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_wikitext_de \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Variable batch size training
CUDA_VISIBLE_DEVICES=0 python ./run_variable_batch_size_lm.py \
  --experiment_config=./configs/wikitext_de_random_init.json \
  --output_dir=./outputs/wikitext_de_variable_batch_size_random_init_et_denominator \
  --project_name=CLP_wikitext_de \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --set_embedding_tuning_denominator \
  --per_device_embedding_tuning_batch_size=1

# Baseline CLP (0% embedding tuning, only full model training)
accelerate launch ./run_language_modeling.py \
  --experiment_config=./configs/oscar_de_random_init.json \
  --output_dir=./outputs/oscar_de_baseline_random_init \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_oscar_de \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=0 \
  --embedding_tuning_percentage=0 \
  --learning_rate=3e-4 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup" \
  --calculate_warmup_based_on_num_train_steps

# Embedding Tuning (only train the embedding layer at the start with a high learning rate)
accelerate launch ./run_language_modeling.py \
  --experiment_config=./configs/oscar_de_random_init.json \
  --output_dir=./outputs/oscar_de_embedding_tuning_random_init \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_oscar_de \
  --per_device_train_batch_size=8 \
  --per_device_eval_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Variable batch size training
CUDA_VISIBLE_DEVICES=0 python ./run_variable_batch_size_lm.py \
  --experiment_config=./configs/oscar_de_random_init.json \
  --output_dir=./outputs/oscar_de_variable_batch_size_random_init_et_denominator \
  --project_name=CLP_wikitext_de \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --set_embedding_tuning_denominator \
  --per_device_embedding_tuning_batch_size=1