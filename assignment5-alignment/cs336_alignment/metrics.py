"""Utilities for parsing model outputs used by evaluation metrics."""

from typing import Any
import re

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """从模型输出中解析 MMLU 预测的选项字母。

    无法唯一确定模型选择了哪个选项时返回 ``None``。

    Args:
        mmlu_example: 包含 ``options`` 等字段的单条 MMLU 样本。
        model_output: 模型针对该样本生成的文本，根据prompt，标准输出是"The correct answer is _",

    Returns:
        解析成功时返回选项字母（如 ``"A"``），否则返回 ``None``。
    """
    options: list[str] = mmlu_example["options"]
    option_letters = tuple(chr(ord("A") + index) for index in range(len(options)))

    # 先统一首尾空白和连续空白 split()默认按任何空格/换行
    output = " ".join(model_output.strip().split())

    if not output:
        return None 
    # 1. 标准回答格式
    pattern = (
        rf"(?:correct answer|answer)\s*(?:is|:)\s*"
        rf"\(?([{''.join(option_letters)}])\)?\b"
    )
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. 只有选项字母，例如 B / B. / (B)
    pattern = rf"\(?([{''.join(option_letters)}])\)?[.:]?"
    match = re.fullmatch(pattern, output, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. 直接输出选项文本
    matches = [
        letter
        for letter, option in zip(option_letters, options)
        if output.casefold() == " ".join(option.strip().split()).casefold()
    ]

    if len(matches) == 1:
        return matches[0]

    return None