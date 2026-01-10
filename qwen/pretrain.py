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
    eval_samples: Optional[int] = field(
        default=2000, metadata={"help": "Streaming 模式下验证集样本数（take 前 N 条）"}
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


logger.info(f"开始加载数据集: {data_args.train_files}")
raw_datasets = load_dataset("json", data_files=data_args.train_files, split="train", streaming=True)
logger.info("数据集加载完成（流式模式）")


# 数据预处理函数
def tokenize_function(examples):
    """对文本进行 tokenize"""
    output = tokenizer(examples["text"],truncation=False,return_attention_mask=True)
    return output


# 对数据集进行 tokenize（流式模式不支持 num_proc）
logger.info("开始对数据集进行 tokenize")
column_names = list(raw_datasets.features)
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
)
logger.info("Tokenize 完成")


# 将文本拼接成固定长度的文本段
def stream_group_texts(dataset, block_size: int):
    """
    将 streaming tokenized dataset 拼接成固定 block_size，并生成 labels
    返回一个新的 iterable dataset
    """
    buffer = {
        "input_ids": [],
        "attention_mask": [],
    }

    def gen():
        nonlocal buffer
        for ex in dataset:
            # ex: {'input_ids': [...], 'attention_mask': [...]}
            buffer["input_ids"].extend(ex["input_ids"])
            buffer["attention_mask"].extend(ex["attention_mask"])

            while len(buffer["input_ids"]) >= block_size:
                input_ids = buffer["input_ids"][:block_size]
                attention_mask = buffer["attention_mask"][:block_size]

                buffer["input_ids"] = buffer["input_ids"][block_size:]
                buffer["attention_mask"] = buffer["attention_mask"][block_size:]

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.copy(),  # CLM
                }

    return datasets.IterableDataset.from_generator(gen)



# 批量处理
logger.info(f"开始将文本拼接成固定长度的块 (block_size={data_args.block_size})")
lm_datasets = stream_group_texts(tokenized_datasets, block_size=data_args.block_size)

# 切分训练集和验证集（Streaming：用 take/skip）
eval_dataset = None
train_dataset = lm_datasets

if data_args.validation_split_percentage and data_args.validation_split_percentage > 0:
    # Streaming 下无法按百分比切分，改为固定验证条数
    eval_samples = 2000  # ✅ 你也可以改成参数
    logger.info(f"Streaming 验证集：take 前 {eval_samples} 个样本作为 eval")
    eval_dataset = lm_datasets.take(eval_samples)
    train_dataset = lm_datasets.skip(eval_samples)
else:
    logger.info("Streaming 模式：不使用验证集")

logger.info("数据集准备完成（Streaming）")


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
