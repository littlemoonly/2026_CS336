"""Datasets and data-loading utilities for alignment training."""

from __future__ import annotations

import json
import random
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from xopen import xopen


ALPACA_SFT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts_safety/alpaca_sft.prompt"
)


class PackedSFTDataset(Dataset):
    """将 instruction-tuning 文档拼接为定长 causal-LM 训练样本。"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | PathLike[str],
        seq_length: int,
        shuffle: bool,
    ) -> None:
        self.seq_length = seq_length

        # xopen 同时支持测试用的 .jsonl 和正式训练数据的 .jsonl.gz。
        with xopen(dataset_path, mode="rt") as file:
            # 每条记录包含 prompt 和 response。
            records: list[dict[str, Any]] = [json.loads(line) for line in file]

        # 必须在拼接文档之前 shuffle，否则只是打乱已经切好的训练块。
        if shuffle:
            random.shuffle(records)

        # 模板文件末尾的换行不属于训练文本，避免它与最后一个 token 合并。
        prompt_template = ALPACA_SFT_PROMPT_PATH.read_text(encoding="utf-8").strip()
        documents = [
            prompt_template.format(
                instruction=record["prompt"],
                response=record["response"],
            )
            for record in records
        ]
        self.token_ids = self._tokenize_and_concatenate(tokenizer, documents)

    @staticmethod
    def _tokenize_and_concatenate(
        tokenizer: PreTrainedTokenizerBase,
        documents: list[str],
    ) -> torch.Tensor:
        """编码每篇文档并拼接成形状 ``(num_tokens,)`` 的 token 流。"""
        token_ids: list[int] = []
        for document in documents:
            # encode 返回 list[int]；该 tokenizer 会为每篇文档添加 BOS。
            document_token_ids = tokenizer.encode(document, add_special_tokens=True)
            document_token_ids.append(tokenizer.eos_token_id)
            token_ids.extend(document_token_ids)

        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        """返回能够生成的完整定长序列数量。"""
        return (self.token_ids.numel() - 1) // self.seq_length

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """返回第 i 个样本；两个张量形状均为 ``(seq_length,)``。"""
        if not 0 <= i < len(self):
            raise IndexError(i)

        start = i * self.seq_length
        input_ids = self.token_ids[start : start + self.seq_length]
        labels = self.token_ids[start + 1 : start + self.seq_length + 1]
        return {"input_ids": input_ids, "labels": labels}


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """返回覆盖 dataset 一个 epoch 的 mini-batch 迭代器。

    默认 collate 会将每条样本中的 ``input_ids`` 和 ``labels`` 分别堆叠；
    最后一个不足 ``batch_size`` 的 batch 也需要保留。
    """
    # TODO(STUDENT): 创建并返回 DataLoader。
    # 传入 dataset、batch_size 和 shuffle，并确保不要丢弃最后一个 batch。
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)