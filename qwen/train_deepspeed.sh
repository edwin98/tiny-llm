#!/bin/bash

# DeepSpeed 预训练示例脚本 - ZeRO Stage 2

deepspeed --num_gpus=2 pretrain.py \
    --deepspeed ds_config_zero2.json \
    --config_name qwen2.5-1.5B \
    --model_name_or_path qwen2.5-1.5B \
    --train_files /root/autodl-tmp/mobvoi_seq_monkey_general_open_corpus.jsonl \
    --block_size 2048 \
    --save_only_model True \
    --output_dir output/pretrain_with_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --max_steps 10000 \
    --save_steps 500 \
    --save_total_limit 1 \
    --validation_split True \
    --eval_strategy steps \
    --eval_steps 500 \
    --eval_accumulation_steps 10 \
    --eval_samples 100000 \
    --load_best_model_at_end False \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --bf16 \
    --gradient_checkpointing \
    --report_to swanlab \
    --ddp_find_unused_parameters False
