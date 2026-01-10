#!/bin/bash

# DeepSpeed 预训练示例脚本 - ZeRO Stage 2

deepspeed --num_gpus=2 pretrain.py \
    --deepspeed ds_config_zero2.json \
    --config_name qwen2.5-1.5B \
    --model_name_or_path qwen2.5-1.5B \
    --train_files ../dataset/mobvoi_seq_monkey_general_open_corpus.jsonl \
    --validation_split_percentage 5 \
    --block_size 2048 \
    --preprocessing_num_workers 10 \
    --output_dir output/pretrain_deepspeed \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --eval_accumulation_steps 10 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --bf16 \
    --gradient_checkpointing \
    --report_to swanlab \
    --ddp_find_unused_parameters False
