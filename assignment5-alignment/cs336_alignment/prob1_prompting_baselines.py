"""
Evaluate OLMo-2-0425-1B on GSM8K under the three Problem 1 prompt baselines.

This file is intentionally a scaffold: it handles the mechanical evaluation
pipeline and leaves the assignment-facing interpretation/manual audit to you.

Example:
    uv run python -m cs336_alignment.prob1_prompting_baselines \
        --model-id-or-dir OLMo-2-0425-1B \
        --num-examples 50 \
        --output-dir outputs/prob1_prompting_baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import torch

try:
    from .checkpoint import get_model_and_tokenizer
    from .drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
    from .util import get_device
    from .vllm_utils import VLLMServer
except ImportError:
    from checkpoint import get_model_and_tokenizer
    from drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
    from util import get_device
    from vllm_utils import VLLMServer

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "OLMo-2-0425-1B"
DEFAULT_GSM8K_PATH = REPO_ROOT / "data" / "gsm8k" / "test.jsonl"
DEFAULT_PROMPT_DIR = REPO_ROOT / "cs336_alignment" / "prompts"
PROMPT_SPECS = {
    "question_only": {
        "template": "question_only.prompt",
        "reward_fn": question_only_reward_fn,
    },
    "r1_zero": {
        "template": "r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
    },
    "r1_zero_three_shot": {
        "template": "r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
    },
}


@dataclass(frozen=True)
class GSM8KExample:
    idx: int
    question: str
    answer: str
    ground_truth: str


@dataclass(frozen=True)
class EvaluatedGeneration:
    prompt_name: str
    example_idx: int
    question: str
    ground_truth: str
    prompt: str
    response: str
    format_reward: float
    answer_reward: float
    reward: float
    category: str
    manual_actually_correct: bool | None = None
    manual_notes: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id-or-dir", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gsm8k-path", type=Path, default=DEFAULT_GSM8K_PATH)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "prob1_prompting_baselines")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--device", default=None, help="Device for transformers backend, e.g. cuda, cpu, mps.")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-gpu", type=int, default=0)
    parser.add_argument("--vllm-launch-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-examples", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-category", type=int, default=5)
    parser.add_argument(
        "--prompt-name",
        choices=sorted(PROMPT_SPECS),
        action="append",
        help="Evaluate only selected prompt(s). Repeat this flag to select multiple prompts.",
    )
    return parser.parse_args()


def load_gsm8k(path: Path, num_examples: int | None = None) -> list[GSM8KExample]:
    examples: list[GSM8KExample] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            examples.append(
                GSM8KExample(
                    idx=idx,
                    question=row["question"],
                    answer=row["answer"],
                    ground_truth=extract_gsm8k_final_answer(row["answer"]),
                )
            )
            if num_examples is not None and len(examples) >= num_examples:
                break
    return examples


def extract_gsm8k_final_answer(answer: str) -> str:
    """Extract the target answer after GSM8K's final-answer marker."""
    marker = "####"
    if marker not in answer:
        raise ValueError(f"GSM8K answer is missing final-answer marker {marker!r}: {answer[:200]}")
    return answer.rsplit(marker, maxsplit=1)[-1].strip().replace(",", "")


def load_prompt_templates(prompt_dir: Path, prompt_names: list[str]) -> dict[str, str]:
    templates = {}
    for prompt_name in prompt_names:
        template_path = prompt_dir / PROMPT_SPECS[prompt_name]["template"]
        templates[prompt_name] = template_path.read_text()
    return templates


def render_prompt(template: str, question: str) -> str:
    return template.format(question=question)


def category_from_rewards(format_reward: float, answer_reward: float) -> str:
    if format_reward == 1.0 and answer_reward == 1.0:
        return "correct_format_and_correct"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "format_correct_answer_wrong"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "format_wrong_answer_wrong"
    raise ValueError(f"Unexpected reward pair: format={format_reward}, answer={answer_reward}")


class TransformersGenerator:
    def __init__(
        self,
        model_id_or_dir: str,
        device: str | None,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> None:
        torch.manual_seed(seed)
        actual_device = get_device(device)
        self.model, self.tokenizer = get_model_and_tokenizer(
            model_id_or_dir=model_id_or_dir,
            device=actual_device,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def __call__(self, prompts: list[str]) -> list[str]:
        responses: list[str] = []
        do_sample = self.temperature > 0.0
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]
            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = self.temperature
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
            for row_ids in output_ids:
                completion_ids = row_ids[prompt_length:]
                responses.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
            logger.info("Generated %d/%d completions", len(responses), len(prompts))
        return responses


class VLLMGenerator:
    def __init__(
        self,
        model_id_or_dir: str,
        host: str,
        port: int,
        gpu: int,
        launch_server: bool,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> None:
        self.server = VLLMServer(
            model_id=model_id_or_dir,
            host=host,
            port=port,
            gpu=gpu,
            seed=seed,
            launch_server=launch_server,
        )
        self.server.start()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed

    def __call__(self, prompts: list[str]) -> list[str]:
        completions = self.server.generate_completions(
            prompts=prompts,
            sampling_params={
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
                "n": 1,
                "seed": self.seed,
            },
            batch_size=self.batch_size,
        )
        return [completion.text for completion in completions]

    def close(self) -> None:
        self.server.stop()


def evaluate_prompt(
    prompt_name: str,
    template: str,
    examples: list[GSM8KExample],
    generate_fn: Callable[[list[str]], list[str]], # 多一个[]?
) -> list[EvaluatedGeneration]:
    prompts = [render_prompt(template, example.question) for example in examples]
    responses = generate_fn(prompts)
    if len(responses) != len(examples):
        raise RuntimeError(f"Expected {len(examples)} responses, got {len(responses)}.")

    reward_fn = PROMPT_SPECS[prompt_name]["reward_fn"]
    evaluated: list[EvaluatedGeneration] = []
    for example, prompt, response in zip(examples, prompts, responses):
        rewards = reward_fn(response=response, ground_truth=example.ground_truth)
        evaluated.append(
            EvaluatedGeneration(
                prompt_name=prompt_name,
                example_idx=example.idx,
                question=example.question,
                ground_truth=example.ground_truth,
                prompt=prompt,
                response=response,
                format_reward=float(rewards["format_reward"]),
                answer_reward=float(rewards["answer_reward"]),
                reward=float(rewards["reward"]),
                category=category_from_rewards(
                    float(rewards["format_reward"]),
                    float(rewards["answer_reward"]),
                ),
                # TODO(STUDENT): During your manual audit, set this to true/false
                # for at least 10 examples from each non-perfect category you inspect.
                manual_actually_correct=None,
                manual_notes=None,
            )
        )
    return evaluated
generate_completions

def summarize(evaluated_by_prompt: dict[str, list[EvaluatedGeneration]]) -> dict:
    summary = {}
    for prompt_name, rows in evaluated_by_prompt.items():
        category_counts = Counter(row.category for row in rows)
        summary[prompt_name] = {
            "n": len(rows),
            "mean_reward": mean(row.reward for row in rows) if rows else 0.0,
            "mean_format_reward": mean(row.format_reward for row in rows) if rows else 0.0,
            "mean_answer_reward": mean(row.answer_reward for row in rows) if rows else 0.0,
            "category_counts": dict(category_counts),
            # TODO(STUDENT): After reading saved examples/output JSONL, report:
            # - among at least 10 category-2 examples, how many are actually correct
            #   but not parsed/rewarded properly?
            # - among at least 10 category-3 examples, how many are actually correct
            #   but not parsed/rewarded properly?
            "manual_audit_notes": "TODO(STUDENT): fill in after inspecting generated outputs.",
        }
    return summary


def write_jsonl(path: Path, rows: list[EvaluatedGeneration]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_review_examples(
    path: Path,
    evaluated_by_prompt: dict[str, list[EvaluatedGeneration]],
    examples_per_category: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    review_payload = {}
    for prompt_name, rows in evaluated_by_prompt.items():
        rows_by_category: dict[str, list[EvaluatedGeneration]] = defaultdict(list)
        for row in rows:
            rows_by_category[row.category].append(row)
        review_payload[prompt_name] = {}
        for category, category_rows in sorted(rows_by_category.items()):
            sampled = list(category_rows)
            rng.shuffle(sampled)
            review_payload[prompt_name][category] = [
                asdict(row) for row in sampled[:examples_per_category]
            ]
    path.write_text(json.dumps(review_payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("Running %s", " ".join(sys.argv))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_names = args.prompt_name or list(PROMPT_SPECS)   # list of str
    examples = load_gsm8k(args.gsm8k_path, args.num_examples)
    templates = load_prompt_templates(args.prompt_dir, prompt_names)
    logger.info("Loaded %d GSM8K examples", len(examples))

    if args.backend == "transformers":
        generator = TransformersGenerator(
            model_id_or_dir=args.model_id_or_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    else:
        generator = VLLMGenerator(
            model_id_or_dir=args.model_id_or_dir,
            host=args.vllm_host,
            port=args.vllm_port,
            gpu=args.vllm_gpu,
            launch_server=args.vllm_launch_server,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )

    evaluated_by_prompt: dict[str, list[EvaluatedGeneration]] = {}
    try:
        for prompt_name in prompt_names:
            logger.info("Evaluating prompt: %s", prompt_name)
            rows = evaluate_prompt(
                prompt_name=prompt_name,
                template=templates[prompt_name],
                examples=examples,
                generate_fn=generator,
            )
            evaluated_by_prompt[prompt_name] = rows
            write_jsonl(args.output_dir / f"{prompt_name}.jsonl", rows)
    finally:
        if isinstance(generator, VLLMGenerator):
            generator.close()

    summary = summarize(evaluated_by_prompt)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_review_examples(
        args.output_dir / "review_examples.json",
        evaluated_by_prompt=evaluated_by_prompt,
        examples_per_category=args.examples_per_category,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote detailed outputs to: {args.output_dir}")
    print("TODO(STUDENT): inspect review_examples.json and the per-prompt JSONL files, then write commentary.")


if __name__ == "__main__":
    main()
