"""Fine-tune Llama 3.1 8B on packed instruction-tuning data."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from cs336_alignment.data import PackedSFTDataset, iterate_batches


logger = logging.getLogger(__name__)

SHARED_ROOT = Path("/mnt/cs336-a5-supplement")
RESULTS_ROOT = Path("/mnt/cs336-a5-supplement-results")
DEFAULT_MODEL_PATH = SHARED_ROOT / "models/Meta-Llama-3.1-8B"
DEFAULT_TRAIN_PATH = (
    SHARED_ROOT / "data/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--wandb-project", default="cs336-a5-sft")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置数据打乱和 PyTorch 使用的随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_lm_loss(
    model: PreTrainedModel,
    batch: dict[str, Tensor],
    device: str,
) -> Tensor:
    """计算一个 microbatch 的平均 next-token cross-entropy loss。"""
    # TODO(STUDENT):
    # 1. 将 input_ids 和 labels 移到 device；形状均为 (batch_size, seq_length)。
    # 2. 前向计算 logits；形状为 (batch_size, seq_length, vocab_size)。
    # 3. 展平前两个维度，用 F.cross_entropy 计算 scalar mean loss。
    # Dataset 已经将 labels 右移过一位，这里不要再次 shift。
    raise NotImplementedError


def train_one_epoch(
    model: PreTrainedModel,
    data_loader: DataLoader,
    optimizer: Optimizer,
    device: str,
    gradient_accumulation_steps: int,
    epoch: int,
    first_update_step: int,
    log_every: int,
    wandb_run: Any,
) -> int:
    """训练一个 epoch，并返回训练结束后的全局 optimizer update step。"""
    # TODO(STUDENT): 实现梯度累积训练循环：
    # 1. model.train()，并在循环前清空梯度；
    # 2. 对每个 microbatch 调用 compute_lm_loss；
    # 3. loss 除以当前累积窗口的 microbatch 数后 backward；完整窗口就是
    #    gradient_accumulation_steps，epoch 尾部可能更少；
    # 4. 每积累指定步数后 optimizer.step() 并 zero_grad()；
    # 5. 最后不足 accumulation steps 的 microbatches 也要完成一次更新；
    # 6. 每 log_every 个 optimizer update，用 wandb_run.log 和 logger.info
    #    记录未缩放的 train/loss、epoch 和 update_step。
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    import wandb

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=config,
        mode=args.wandb_mode,
    )
    wandb_run.define_metric("update_step")
    wandb_run.define_metric("train/*", step_metric="update_step")

    try:
        logger.info("Loading tokenizer and model from %s", args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(args.device)
        model.config.use_cache = False

        train_dataset = PackedSFTDataset(
            tokenizer=tokenizer,
            dataset_path=args.train_path,
            seq_length=args.seq_length,
            shuffle=args.shuffle,
        )
        train_loader = iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        logger.info(
            "Training on %d sequences; microbatch=%d, effective_batch=%d",
            len(train_dataset),
            args.batch_size,
            args.batch_size * args.gradient_accumulation_steps,
        )

        update_step = 0
        for epoch in range(1, args.epochs + 1):
            update_step = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=args.device,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                epoch=epoch,
                first_update_step=update_step,
                log_every=args.log_every,
                wandb_run=wandb_run,
            )

        logger.info("Saving model and tokenizer to %s", args.output_dir)
        model.save_pretrained(save_directory=args.output_dir)
        tokenizer.save_pretrained(save_directory=args.output_dir)
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
