# 预训练使用说明

## 主要更新

### 1. 验证集支持
- 添加 `validation_split_percentage` 参数，默认切分 5% 作为验证集
- 支持在训练过程中定期评估模型性能
- 自动保存最佳 checkpoint

### 2. 监控指标
- **Loss**: 训练和验证损失
- **Perplexity**: 困惑度（自动从 loss 计算）
- **Learning Rate**: 学习率变化
- **Gradient Norm**: 梯度范数（需在 TrainingArguments 中启用）

### 3. DeepSpeed 支持
- 提供三种 DeepSpeed 配置：
  - `ds_config_zero2.json`: ZeRO Stage 2（推荐）
  - `ds_config_zero3.json`: ZeRO Stage 3（大模型）
  - `ds_config_zero3_offload.json`: ZeRO Stage 3 + CPU Offload（显存不足时）

## 使用方法

### 基础训练（单卡）

```bash
python pretrain.py \
    --model_name_or_path qwen2.5-1.5B \
    --train_files ../dataset/your_data.jsonl \
    --validation_split_percentage 5 \
    --block_size 2048 \
    --output_dir output/pretrain \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --bf16
```

### DeepSpeed 训练（多卡）

```bash
# ZeRO Stage 2（推荐）
deepspeed --num_gpus=2 pretrain.py \
    --deepspeed ds_config_zero2.json \
    --model_name_or_path qwen2.5-1.5B \
    ...其他参数

# ZeRO Stage 3（大模型）
deepspeed --num_gpus=4 pretrain.py \
    --deepspeed ds_config_zero3.json \
    ...

# ZeRO Stage 3 + Offload（显存不足）
deepspeed --num_gpus=2 pretrain.py \
    --deepspeed ds_config_zero3_offload.json \
    ...
```

## 关键参数说明

### 数据参数
- `--train_files`: 训练数据文件路径（JSON/JSONL 格式）
- `--validation_split_percentage`: 验证集占比（0-100），设为 0 则无验证集
- `--block_size`: 文本块长度，默认 2048
- `--preprocessing_num_workers`: 数据预处理线程数

### 评估参数
- `--evaluation_strategy`: 评估策略（no/steps/epoch）
- `--eval_steps`: 每隔多少步评估一次
- `--eval_accumulation_steps`: 评估时梯度累积步数
- `--load_best_model_at_end`: 训练结束后加载最佳模型
- `--metric_for_best_model`: 用于选择最佳模型的指标（eval_loss）
- `--greater_is_better`: 指标越大越好还是越小越好

### 训练参数
- `--per_device_train_batch_size`: 每个设备的 batch size
- `--gradient_accumulation_steps`: 梯度累积步数
- `--learning_rate`: 学习率
- `--num_train_epochs`: 训练轮数
- `--bf16`: 使用 bfloat16 混合精度（推荐）
- `--gradient_checkpointing`: 梯度检查点（节省显存）

### 保存参数
- `--output_dir`: 输出目录
- `--save_steps`: 每隔多少步保存一次
- `--save_total_limit`: 最多保留多少个 checkpoint

## 监控训练

### TensorBoard

```bash
tensorboard --logdir output/pretrain/runs
```

### SwanLab
训练会自动记录到 SwanLab，访问 https://swanlab.cn 查看

## 训练建议

1. **验证集占比**:
   - 数据量小（< 10万条）：5-10%
   - 数据量大（> 100万条）：1-3%

2. **评估频率**:
   - 快速迭代：每 100-500 步
   - 稳定训练：每 1000-2000 步

3. **保存策略**:
   - 设置 `save_total_limit=3` 只保留最近 3 个 checkpoint
   - 启用 `load_best_model_at_end` 自动加载最佳模型

4. **DeepSpeed 选择**:
   - 小模型（< 3B）: ZeRO Stage 2
   - 大模型（7B-13B）: ZeRO Stage 3
   - 显存不足: ZeRO Stage 3 + Offload

## 监控指标解读

- **Loss**: 应该持续下降，如果波动大可能学习率过高
- **Perplexity**: 越低越好，通常在 10-100 之间
- **Eval Loss vs Train Loss**:
  - 如果 eval_loss > train_loss 且差距持续增大 → 过拟合
  - 解决方法：增加数据、减少 epoch、添加正则化
