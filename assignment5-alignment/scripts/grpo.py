"""Run standard on-policy GRPO on GSM8K with a vLLM rollout server."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.util import grpo_train_step
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Example:
    idx: int
    question: str
    ground_truth: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id-or-dir", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--prompt-path", type=Path, default=REPO_ROOT / "cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--train-path", type=Path, default=REPO_ROOT / "data/gsm8k/train.jsonl")
    parser.add_argument("--val-path", type=Path, default=REPO_ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/grpo")

    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)   # 共训练200步？
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256, help="Number of responses, not prompts.")
    parser.add_argument("--train-batch-size", type=int, default=256, help="Number of responses, not prompts.")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--logged-val-rollouts", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=128, help="Prompts per vLLM request.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--policy-dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--compute-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--vllm-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-launch-server", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--wandb-project", default="cs336-a5-grpo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="grpo-standard-on-policy")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--save-final", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rollout_batch_size != args.train_batch_size:
        raise ValueError("Standard on-policy GRPO requires rollout_batch_size == train_batch_size.")
    if args.rollout_batch_size % args.group_size:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    if args.train_batch_size % args.gradient_accumulation_steps:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    if args.n_train_examples < args.rollout_batch_size // args.group_size:
        raise ValueError("n_train_examples must contain at least one prompt batch.")
    if min(args.n_val_examples, args.num_rollout_steps, args.eval_every) <= 0:
        raise ValueError("n_val_examples, num_rollout_steps, and eval_every must be positive.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_answer(answer: str) -> str:
    if "####" not in answer:
        raise ValueError("GSM8K answer is missing the final-answer marker '####'.")
    return answer.rsplit("####", maxsplit=1)[-1].strip().replace(",", "")


def load_examples(path: Path, limit: int, rng: random.Random | None = None) -> list[Example]:
    examples = []
    with path.open() as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            examples.append(Example(idx, row["question"], extract_answer(row["answer"])))
    if limit > len(examples):
        raise ValueError(f"Requested {limit} examples from {path}, which contains {len(examples)}.")
    if rng is not None:
        rng.shuffle(examples)
    return examples[:limit]


def render_prompts(template: str, examples: list[Example]) -> list[str]:
    return [template.format(question=example.question) for example in examples]


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item()
    return value


def score_completions(completions: list[VLLMCompletion], examples: list[Example]) -> dict[str, float]:
    totals = {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    response_tokens = 0
    for completion, example in zip(completions, examples, strict=True):
        rewards = r1_zero_reward_fn(completion.text, example.ground_truth)
        for key in totals:
            totals[key] += float(rewards[key])
        response_tokens += len(completion.token_ids)
    count = len(examples)
    return {
        "reward": totals["reward"] / count,
        "format_reward": totals["format_reward"] / count,
        "answer_reward": totals["answer_reward"] / count,
        "response_length": response_tokens / count,
    }


def rollout_rows(
    split: str,
    step: int,
    examples: list[Example],
    completions: list[VLLMCompletion],
) -> list[dict[str, Any]]:
    rows = []
    for example, completion in zip(examples, completions, strict=True):
        rewards = r1_zero_reward_fn(completion.text, example.ground_truth)
        rows.append(
            {
                "split": split,
                "step": step,
                "example_idx": example.idx,
                "question": example.question,
                "ground_truth": example.ground_truth,
                "response": completion.text,
                "response_length": len(completion.token_ids),
                **{key: float(value) for key, value in rewards.items()},
            }
        )
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wandb_table(wandb: Any, rows: list[dict[str, Any]]) -> Any:
    columns = list(rows[0])
    return wandb.Table(columns=columns, data=[[row[column] for column in columns] for row in rows])


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Command: %s", " ".join(sys.argv))
    set_seed(args.seed)

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    rollouts_path = run_dir / "rollouts.jsonl"
    metrics_path.write_text("")
    rollouts_path.write_text("")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    import wandb

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name or f"seed-{args.seed}",
        config=config,
        mode=args.wandb_mode,
    )
    wandb_run.define_metric("step")
    wandb_run.define_metric("train/*", step_metric="step")
    wandb_run.define_metric("val/*", step_metric="step")

    rng = random.Random(args.seed)
    train_examples = load_examples(args.train_path, args.n_train_examples, rng)
    val_examples = load_examples(args.val_path, args.n_val_examples)
    prompt_template = args.prompt_path.read_text()
    prompts_per_step = args.rollout_batch_size // args.group_size
    train_order = list(range(len(train_examples)))
    rng.shuffle(train_order)
    train_cursor = 0

    server = VLLMServer(
        model_id=args.model_id_or_dir,
        host=args.vllm_host,
        port=args.vllm_port,
        gpu=args.vllm_gpu,
        seed=args.seed,
        dtype=args.vllm_dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enforce_eager=args.vllm_enforce_eager,
        launch_server=args.vllm_launch_server,
    )

    try:
        server.start()
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        policy, tokenizer = get_model_and_tokenizer(
            args.model_id_or_dir,
            args.policy_device,
            torch_dtype=dtypes[args.policy_dtype],
            attn_implementation="sdpa",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if args.gradient_checkpointing:
            policy.gradient_checkpointing_enable()
            policy.config.use_cache = False
        policy.train()
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            foreach=False,
        )
        grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=args.compute_dtype == "float16",
        )
        server.init_weight_sync(args.policy_device)
        synced_step = -1

        def sync_policy(step: int) -> None:
            nonlocal synced_step
            if synced_step != step:
                logger.info("Syncing policy weights for step %d", step)
                server.sync_policy_weights(policy)
                synced_step = step

        def evaluate(step: int) -> None:
            sync_policy(step)
            prompts = render_prompts(prompt_template, val_examples)
            completions = server.generate_completions(
                prompts,
                {
                    "temperature": args.sampling_temperature,
                    "max_tokens": args.sampling_max_tokens,
                    "n": 1,
                    "seed": args.seed + 100_000 + step,
                },
                batch_size=args.generation_batch_size,
            )
            metrics = {f"val/{key}": value for key, value in score_completions(completions, val_examples).items()}
            metrics["step"] = step
            append_jsonl(metrics_path, [metrics])
            sampled_rows = rollout_rows("val", step, val_examples, completions)[: args.logged_val_rollouts]
            append_jsonl(rollouts_path, sampled_rows)
            wandb_run.log({**metrics, "val/rollouts": wandb_table(wandb, sampled_rows)})
            logger.info("step=%d val_reward=%.4f val_format=%.4f", step, metrics["val/reward"], metrics["val/format_reward"])

        evaluate(0)
        for step in range(1, args.num_rollout_steps + 1):
            step_started_at = time.perf_counter()
            # 先取 32 个不同的 GSM8K question
            if train_cursor + prompts_per_step > len(train_order):
                rng.shuffle(train_order)
                train_cursor = 0
            batch_indices = train_order[train_cursor : train_cursor + prompts_per_step]
            train_cursor += prompts_per_step
            prompt_examples = [train_examples[index] for index in batch_indices]

            sync_policy(step - 1)   # 把上一轮训练完成后的最新 policy 参数同步给 vLLM
            prompts = render_prompts(prompt_template, prompt_examples)
            # vLLM rollout, 每个 prompt 生成 8 个 completion
            rollout_started_at = time.perf_counter()
            completions = server.generate_completions(
                prompts,
                {
                    "temperature": args.sampling_temperature,
                    "max_tokens": args.sampling_max_tokens,
                    "n": args.group_size,
                    "seed": args.seed + step,
                },
                batch_size=args.generation_batch_size,
            )
            if len(completions) != args.rollout_batch_size:
                raise RuntimeError(f"Expected {args.rollout_batch_size} rollouts, got {len(completions)}.")
            # 把 prompt 和答案复制成与 256 个 rollout 一一对应的形式
            repeated_examples = [example for example in prompt_examples for _ in range(args.group_size)]
            repeated_prompts = [prompt for prompt in prompts for _ in range(args.group_size)]
            responses = [completion.text for completion in completions]
            ground_truths = [example.ground_truth for example in repeated_examples]
            rollout_seconds = time.perf_counter() - rollout_started_at
            torch.cuda.reset_peak_memory_stats(torch.device(args.policy_device))
            train_started_at = time.perf_counter()
            loss, train_metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=responses,
                repeated_ground_truths=ground_truths,
                group_size=args.group_size,
                autocast_dtype=dtypes[args.compute_dtype],
                grad_scaler=grad_scaler,
            )
            train_seconds = time.perf_counter() - train_started_at
            train_metrics = {
                "train/loss": scalar(loss),
                "train/grad_norm": scalar(train_metadata["grad_norm_before_clipping"]),
                "train/token_entropy": scalar(train_metadata["token_entropy"]),
                "train/reward": scalar(train_metadata["reward_mean"]),
                "train/format_reward": scalar(train_metadata["format_reward_mean"]),
                "train/answer_reward": scalar(train_metadata["answer_reward_mean"]),
                "train/reward_std": scalar(train_metadata["reward_std"]),
                "train/response_length": scalar(train_metadata["response_length"]),
                "train/learning_rate": args.learning_rate,
                "train/rollout_seconds": rollout_seconds,
                "train/update_seconds": train_seconds,
                "train/step_seconds": time.perf_counter() - step_started_at,
                "train/gpu_peak_memory_gb": torch.cuda.max_memory_allocated(
                    torch.device(args.policy_device)
                )
                / 2**30,
                "step": step,
            }
            append_jsonl(metrics_path, [train_metrics])
            log_payload: dict[str, Any] = dict(train_metrics)

            if step == 1 or step % args.log_rollouts_every == 0 or step == args.num_rollout_steps:
                rows = rollout_rows("train", step, repeated_examples, completions)
                append_jsonl(rollouts_path, rows)
                log_payload["train/rollouts"] = wandb_table(wandb, rows)
            wandb_run.log(log_payload)
            logger.info(
                "step=%d loss=%.5f reward=%.4f entropy=%.4f grad_norm=%.4f memory=%.2fGB time=%.1fs",
                step,
                train_metrics["train/loss"],
                train_metrics["train/reward"],
                train_metrics["train/token_entropy"],
                train_metrics["train/grad_norm"],
                train_metrics["train/gpu_peak_memory_gb"],
                train_metrics["train/step_seconds"],
            )

            if step % args.eval_every == 0 or step == args.num_rollout_steps:
                evaluate(step)

        if args.save_final:
            checkpoint_dir = run_dir / "checkpoint-final"
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
    finally:
        server.stop()
        wandb_run.finish()


if __name__ == "__main__":
    main()
