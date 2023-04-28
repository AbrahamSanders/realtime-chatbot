python train.py \
    --log_level info \
    --model_name_or_path=facebook/opt-2.7b \
    --no_use_fast_tokenizer \
    --train_file=data/dataset_train.txt \
    --validation_file=data/dataset_dev.txt \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --do_train \
    --gradient_accumulation_steps=32 \
    --output_dir=rtchat-2.7b \
    --do_eval \
    --overwrite_output_dir \
    --seed=42 \
    --data_seed=42 \
    --eval_steps=163 \
    --logging_steps=10 \
    --save_total_limit=2 \
    --evaluation_strategy=steps \
    --lr_scheduler_type=linear \
    --num_train_epochs=4 \
    --save_steps=163 \
    --learning_rate=5e-05 \
    --warmup_ratio=0.1 \
    --metric_for_best_model=eval_loss \
    --load_best_model_at_end \
    --dataloader_drop_last \
    --gradient_checkpointing \
    --use_anchor_model \
    --anchor_loss_weight=0.5 \
    --kl_div_temperature=1.0 \
    --embed_cosine_loss_weight=0.5