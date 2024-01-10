# Embedding Tuning
accelerate launch ../run_language_modeling.py \
  --experiment_config=../configs/oscar_de_experiments.json \
  --output_dir=../outputs/oscar_de_embedding_tuning \
  --preprocessed_dataset_path=../data/oscar_de/preprocessed \
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