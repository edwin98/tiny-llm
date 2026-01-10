import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import swanlab
import torch
import transformers
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="qwen2.5-1.5B",
        metadata={"help": "预训练模型的路径或模型标识符"},
    )

@dataclass
class DataTrainingArguments:
    train_files: str = field(
        default=None,
        metadata={"help": "训练数据文件路径"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "最大训练样本数量，None表示使用全部数据"},
    )


def preprocess(
    sources,
    tokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
):
    """
    预处理多轮对话数据

    Args:
        sources: 多轮对话数据列表
        tokenizer: 分词器
        max_len: 最大序列长度
        system_message: 系统提示词
    """
    roles = {
        "human": "<|im_start|>human",
        "assistant": "<|im_start|>assistant",
    }

    # 不同的 tokenizer 需要特别定义
    # BOS
    im_start = tokenizer("<|im_start|>").input_ids
    # EOS
    im_end = tokenizer("<|im_end|>").input_ids
    # PAD
    IGNORE_TOKEN_ID = -100
    # 换行符
    nl_tokens = tokenizer("\n").input_ids
    # 角色标识符
    _system = tokenizer("system").input_ids + nl_tokens

    # 拼接多轮对话
    input_ids, targets = [], []
    # 多个样本
    for i in tqdm(range(len(sources)), desc="处理数据"):
        # source 为一个多轮对话样本
        source = sources[i]
        # 从 user 开始
        if source[0]["from"] != "human":
            source = source[1:]
        # 分别是输入和输出
        input_id, target = [], []
        # system: 【BOS】system\nYou are a helpful assistant.【EOS】\n
        system = (
            im_start
            + _system
            + tokenizer(system_message).input_ids
            + im_end
            + nl_tokens
        )
        input_id += system
        # system 不需要拟合
        target += im_start + [IGNORE_TOKEN_ID] * (len(system) - 3) + im_end + nl_tokens
        assert len(input_id) == len(target)
        # 依次拼接
        for j, sentence in enumerate(source):
            # sentence 为一轮对话
            role = roles[sentence["from"]]
            # user：<|im_start|>human\ninstruction【EOS】\n
            # assistant：<|im_start|>assistant\nresponse【EOS】\n
            _input_id = (
                tokenizer(role).input_ids
                + nl_tokens
                + tokenizer(sentence["value"]).input_ids
                + im_end
                + nl_tokens
            )
            input_id += _input_id
            if role == "<|im_start|>human":
                # user 不需要拟合
                _target = (
                    im_start
                    + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                    + im_end
                    + nl_tokens
                )
            elif role == "<|im_start|>assistant":
                # assistant 需要拟合
                _target = (
                    im_start
                    + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
                    + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                    + im_end
                    + nl_tokens
                )
            else:
                print(role)
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        # 最后进行 PAD
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids)
    targets = torch.tensor(targets)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(TorchDataset):
    """监督微调数据集"""

    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # 加载并预处理数据
        sources = [example["conversations"] for example in raw_data]
        # preprocess 即上文定义的数据预处理逻辑
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def main():
    # 加载脚本参数
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化 SwanLab
    swanlab.init(project="sft", experiment_name="qwen-1.5b")

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 将日志级别设置为 INFO
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 训练整体情况记录
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查 checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"输出路径 ({training_args.output_dir}) 非空 ")
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(f"从 {last_checkpoint}恢复训练")

    # 设置随机数种子
    set_seed(training_args.seed)

    # 初始化模型
    logger.warning("加载预训练模型")
    logger.info(f"模型参数地址：{model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"继承一个预训练模型 - Total size={n_params / 2**20:.2f}M params")

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    logger.info("完成 tokenizer 加载")

    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载微调数据
    logger.info(f"训练集地址：{data_args.train_files}")
    with open(data_args.train_files, encoding="utf-8") as f:
        all_lines = f.readlines()
        if data_args.max_samples is not None:
            all_lines = all_lines[: data_args.max_samples]
            logger.info(f"限制训练样本数量为: {data_args.max_samples}")
        lst = [json.loads(line) for line in all_lines]
    logger.info("完成训练集加载")
    logger.info(f"训练样本总数:{len(lst)}")

    # 创建数据集
    train_dataset = SupervisedDataset(
        lst, tokenizer=tokenizer, max_len=data_args.max_seq_length
    )

    # 初始化 Trainer
    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # 从 checkpoint 加载
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # 开始训练
    logger.info("开始训练")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # 保存模型
    trainer.save_model()

    # 保存训练结果
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("训练完成")


if __name__ == "__main__":
    main()
