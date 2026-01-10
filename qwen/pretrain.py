import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List
import datetime

import datasets
import numpy as np
import swanlab
import transformers
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint


# 超参类
@dataclass
class ModelArguments:
    """
    关于模型的参数
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("后训练使用，为预训练模型参数地址")},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练使用，Config 文件地址"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练 Tokenizer 地址"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": ("模型训练使用的数据类型，推荐 bfloat16"),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    关于训练的参数
    """

    train_files: Optional[List[str]] = field(
        default=None, metadata={"help": "训练数据路径"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "验证集占比（百分比），默认 5%"},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": ("设置的文本块长度")},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使用线程数."},
    )


# 自定义 Callback：记录 Perplexity
class PerplexityCallback(TrainerCallback):
    """在评估时计算并记录 Perplexity"""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估结束后，计算 perplexity"""
        if metrics is not None and "eval_loss" in metrics:
            try:
                perplexity = np.exp(metrics["eval_loss"])
                metrics["eval_perplexity"] = perplexity
                logger.info(f"Eval Perplexity: {perplexity:.4f}")
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")
                logger.warning("Perplexity overflow!")


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# 设置日志级别（transformers 和 datasets 库的日志）
transformers.utils.logging.set_verbosity_info()
datasets.utils.logging.set_verbosity_info()

# 训练整体情况记录
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# 初始化 SwanLab
swanlab.init(
    project="pretrain",
    experiment_name=f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

# 检查 checkpoint
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    # 使用 transformers 自带的 get_last_checkpoint 自动检测
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(f"输出路径 ({training_args.output_dir}) 非空 ")
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"从 {last_checkpoint}恢复训练")


# 初始化模型
if model_args.config_name is not None:
    # from scrach
    config = AutoConfig.from_pretrained(model_args.config_name)
    logger.warning("你正在从零初始化一个模型")
    logger.info(f"模型参数配置地址：{model_args.config_name}")
    logger.info(f"模型参数：{config}")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"预训练一个新模型 - Total size={n_params / 2**20:.2f}M params")
elif model_args.model_name_or_path is not None:
    logger.warning("你正在初始化一个预训练模型")
    logger.info(f"模型参数地址：{model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"继承一个预训练模型 - Total size={n_params / 2**20:.2f}M params")
else:
    logger.error("config_name 和 model_name_or_path 不能均为空")
    raise ValueError("config_name 和 model_name_or_path 不能均为空")


# 初始化 Tokenizer
if model_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
elif model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
else:
    logger.error("必须提供 tokenizer_name 或 model_name_or_path")
    raise ValueError("必须提供 tokenizer_name 或 model_name_or_path")

logger.info(f"Tokenizer 初始化完成: {tokenizer}")


# 加载数据集
logger.info(f"开始加载数据集: {data_args.train_files}")
raw_datasets = load_dataset("json", data_files=data_args.train_files, split="train")
logger.info(f"数据集加载完成，共 {len(raw_datasets)} 条数据")


# 数据预处理函数
def tokenize_function(examples):
    """对文本进行 tokenize"""
    output = tokenizer(examples["text"])
    return output


# 对数据集进行 tokenize
logger.info("开始对数据集进行 tokenize")
column_names = list(raw_datasets.features)
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)
logger.info(f"Tokenize 完成")


# 将文本拼接成固定长度的文本段
def group_texts(examples):
    """将文本段拼接成固定长度的块"""
    # 获取块长度
    block_size = data_args.block_size
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            logger.warning(
                f"tokenizer 的 model_max_length ({block_size}) 过大，设置为 2048"
            )
            block_size = 2048

    # 将文本段拼接起来
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # 计算拼起来的整体长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 如果长度太长，进行分块
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # 按 block_size 进行切分
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # CLM 任务，labels 和 input 是相同的
    result["labels"] = result["input_ids"].copy()
    return result


# 批量处理
logger.info(f"开始将文本拼接成固定长度的块 (block_size={data_args.block_size})")
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {data_args.block_size}",
)

# 切分训练集和验证集
if data_args.validation_split_percentage > 0:
    logger.info(f"切分验证集，占比 {data_args.validation_split_percentage}%")
    split_dataset = lm_datasets.train_test_split(
        test_size=data_args.validation_split_percentage / 100,
        seed=42,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logger.info(
        f"数据集切分完成 - 训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条"
    )
else:
    train_dataset = lm_datasets
    eval_dataset = None
    logger.info(f"数据处理完成，共 {len(train_dataset)} 个 batch，无验证集")


logger.info("初始化 Trainer")
callbacks = [PerplexityCallback()] if eval_dataset is not None else []
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=default_data_collator,
    callbacks=callbacks,
)

# 从 checkpoint 加载
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
    checkpoint = last_checkpoint

logger.info("开始训练")
train_result = trainer.train(resume_from_checkpoint=checkpoint)
logger.info("训练完成，保存模型")
trainer.save_model()

# 记录训练结果
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# 如果有验证集，进行最终评估
if eval_dataset is not None:
    logger.info("开始最终评估")
    eval_metrics = trainer.evaluate()
    logger.info(f"最终评估结果: {eval_metrics}")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

logger.info("训练流程全部完成")
