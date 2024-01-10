# Baseline CLP (0% embedding tuning, only full model training)
accelerate launch ./run_language_modeling.py \
  --experiment_config=./configs/oscar_de_experiments.json \
  --output_dir=./outputs/oscar_de_baseline \
  --preprocessed_dataset_path=./data/oscar_de/preprocessed \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_oscar_de \
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
  --experiment_config=./configs/oscar_de_experiments.json \
  --output_dir=./outputs/oscar_de_embedding_tuning \
  --preprocessed_dataset_path=./data/oscar_de/preprocessed \
  --preprocessed_dataset_path=./data/oscar_de/preprocessed \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_oscar_de \
  --warmup_percentage=1 \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps
  
# Accelerated Training Start (no freezing/ embedding tuning, only high learning rate at the start)
accelerate launch ./run_language_modeling.py \
  --experiment_config=./configs/oscar_de_experiments.json \
  --output_dir=./outputs/oscar_de_accelerated_training_start \
  --preprocessed_dataset_path=./data/oscar_de/preprocessed \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_oscar_de \
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
accelerate launch ./run_language_modeling.py \
  --experiment_config=./configs/oscar_de_experiments.json \
  --output_dir=./outputs/oscar_de_frozen_transformer_layers \
  --preprocessed_dataset_path=./data/oscar_de/preprocessed \
  --with_tracking \
  --report_to=wandb \
  --project_name=CLP_oscar_de \
  --embedding_tuning_warmup_percentage=1 \
  --embedding_tuning_percentage=100 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --calculate_warmup_based_on_num_train_steps

# Variable batch size training (1 for embedding tuning, 64 for full model training)  
# embedding tuning with custom embedding tuning denominator
CUDA_VISIBLE_DEVICES=0 python ./run_variable_batch_size_lm.py \
  --experiment_config=./configs/oscar_de_embedding_tuning_variable_batch_size.json \
  --output_dir=./outputs/oscar_de_variable_batch_size_et_denominator \
  --preprocessed_dataset_path=./data/oscar_de/preprocessed \
  --project_name=CLP_oscar_de \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --set_embedding_tuning_denominator \
  --per_device_embedding_tuning_batch_size=1
  
# no freezing of transformer layers, custom embedding tuning denominator
CUDA_VISIBLE_DEVICES=0 python ./run_variable_batch_size_lm.py \
  --experiment_config=./configs/oscar_de_embedding_tuning_variable_batch_size.json \
  --output_dir=./outputs/oscar_de_variable_batch_size_accelerated_training_start_et_denominator \
  --preprocessed_dataset_path=./data/oscar_de/preprocessed \
  --project_name=CLP_oscar_de \
  --warmup_percentage=10 \
  --embedding_tuning_warmup_percentage=10 \
  --embedding_tuning_percentage=10 \
  --learning_rate=0.001 \
  --full_training_learning_rate=3e-4 \
  --min_lr=3e-5 \
  --not_freeze_transformer_layers \
  --lr_scheduler_type="cosine_schedule_with_warmup_embedding_tuning" \
  --set_embedding_tuning_denominator \
  --per_device_embedding_tuning_batch_size=1