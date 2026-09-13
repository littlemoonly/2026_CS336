"""Launch instruction fine-tuning on Modal with supplement volumes mounted."""

from __future__ import annotations

from cs336_alignment.modal_utils_safety import (
    BASE_MODEL_PATH,
    RESULTS_VOLUME_MOUNT_PATH,
    SFT_TRAIN_PATH,
    app,
    run_command,
)


@app.local_entrypoint()
def main(
    epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    seq_length: int = 512,
    wandb_mode: str = "online",
) -> None:
    """组装并提交 SFT 远端训练命令。"""
    run_command.remote(
        [
            "python",
            "-u",
            "-m",
            "cs336_alignment.train_sft",
            "--model-name-or-path",
            BASE_MODEL_PATH,
            "--train-path",
            SFT_TRAIN_PATH,
            "--output-dir",
            f"{RESULTS_VOLUME_MOUNT_PATH}/sft",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--gradient-accumulation-steps",
            str(gradient_accumulation_steps),
            "--seq-length",
            str(seq_length),
            "--wandb-mode",
            wandb_mode,
        ]
    )
