#!/bin/bash

# 预训练示例脚本 - 使用验证集和监控指标

python pretrain.py \
    --model_name_or_path qwen2.5-1.5B \
    --config_name qwen2.5-1.5B \
    --train_files /root/autodl-tmp/mobvoi_seq_monkey_general_open_corpus.jsonl \
    --validation_split_percentage 5 \
    --block_size 2048 \
    --preprocessing_num_workers 10 \
    --output_dir output/pretrain_with_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --max_steps 10000 \
    --save_steps 500 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 500 \
    --eval_accumulation_steps 10 \
    --load_best_model_at_end False \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --bf16 \
    --gradient_checkpointing \
    --report_to swanlab \
    --ddp_find_unused_parameters False
