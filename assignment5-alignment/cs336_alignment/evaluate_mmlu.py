"""Evaluate zero-shot MMLU performance and serialize per-example results.

The default paths are suitable for the Assignment 5 supplement Modal image.
Run this module through ``mmlu_modal.py`` so the shared model volume is mounted.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

from cs336_alignment.metrics import parse_mmlu_response


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MMLU_DIR = REPO_ROOT / "data/mmlu/test"
DEFAULT_PROMPT_PATH = REPO_ROOT / "cs336_alignment/prompts_safety/mmlu_zero_shot.prompt"
DEFAULT_MODEL_PATH = Path("/mnt/cs336-a5-supplement/models/Meta-Llama-3.1-8B")
DEFAULT_OUTPUT_DIR = Path("/mnt/cs336-a5-supplement-results/mmlu_zero_shot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlu-dir", type=Path, default=DEFAULT_MMLU_DIR)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_mmlu_examples(
    mmlu_dir: str | Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """读取目录中所有 MMLU CSV，并转换成统一的样本字典。"""
    examples: list[dict[str, Any]] = []
    for csv_path in sorted(Path(mmlu_dir).glob("*_test.csv")):
        subject = csv_path.stem.removesuffix("_test")
        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.reader(file):
                question, *options, answer = row
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": options,
                        "answer": answer,
                    }
                )
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples


def format_mmlu_prompts(
    examples: list[dict[str, Any]],
    prompt_template: str,
) -> list[str]:
    """使用同一个 zero-shot 模板格式化每条 MMLU 样本。"""
    return [prompt_template.format(**example) for example in examples]


def generate_outputs(
    prompts: list[str],
    model_name_or_path: str,
    tensor_parallel_size: int,
    max_tokens: int,
) -> list[str]:
    """使用 vLLM 对 prompts 做确定性的批量生成。"""
    # TODO(STUDENT):
    # 1. 用 model_name_or_path 和 tensor_parallel_size 初始化 vLLM.LLM；
    # 2. 构造 temperature=0 的 SamplingParams，并限制 max_tokens；
    # 3. 批量调用 model.generate；
    # 4. 从每个 RequestOutput 中取出第一个 completion 的 text。
    from vllm import LLM, SamplingParams

    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    request_outputs = model.generate(
        prompts,
        sampling_params
    )
    return [
        request_output.output[0].text.strip() 
        for request_output in request_outputs
    ]


def score_outputs(
    examples: list[dict[str, Any]],
    prompts: list[str],
    model_outputs: list[str],
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """解析生成结果，计算 accuracy，并生成可写入 JSONL 的记录。"""
    assert len(examples) == len(prompts) == len(model_outputs)

    records: list[dict[str, Any]] = []
    for example, prompt, model_output in zip(
        examples, prompts, model_outputs, strict=True
    ):
        prediction = parse_mmlu_response(example, model_output)
        records.append(
            {
                **example,
                "prompt": prompt,
                "model_output": model_output,
                "prediction": prediction,
                "correct": prediction == example["answer"],
            }
        )

    accuracy = mean(record["correct"] for record in records) if records else 0.0
    parsed_count = sum(record["prediction"] is not None for record in records)
    metrics: dict[str, float | int] = {
        "accuracy": accuracy,
        "parse_rate": parsed_count / len(records) if records else 0.0,
        "num_examples": len(records),
        "num_parsed": parsed_count,
    }
    return records, metrics


def serialize_results(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    metrics: dict[str, float | int],
) -> None:
    """将逐样本结果写成 JSONL，并将汇总指标写成 JSON。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    examples = load_mmlu_examples(args.mmlu_dir, args.limit)
    prompt_template = args.prompt_path.read_text(encoding="utf-8")
    '''
    prompt_template: Answer the following multiple choice question about {subject}. Respond with a single
    ↪ sentence of the form "The correct answer is _", filling the blank with the letter
    ↪ corresponding to the correct answer (i.e., A, B, C or D).
    Question: {question}
    A. {options[0]}
    B. {options[1]}
    C. {options[2]}
    D. {options[3]}
    Answer:
    '''
    prompts = format_mmlu_prompts(examples, prompt_template)
    logger.info("Loaded and formatted %d MMLU examples", len(examples))

    model_outputs = generate_outputs(
        prompts=prompts,
        model_name_or_path=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
    )
    records, metrics = score_outputs(examples, prompts, model_outputs)
    serialize_results(args.output_dir, records, metrics)

    logger.info("Metrics: %s", json.dumps(metrics, sort_keys=True))
    logger.info("Wrote results to %s", args.output_dir)


if __name__ == "__main__":
    main()
