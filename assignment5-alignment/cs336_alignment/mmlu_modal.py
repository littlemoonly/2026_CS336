"""Launch the zero-shot MMLU evaluator on Modal with shared volumes mounted."""

from __future__ import annotations

from cs336_alignment.modal_utils_safety import (
    BASE_MODEL_PATH,
    RESULTS_VOLUME_MOUNT_PATH,
    app,
    run_command,
)


@app.local_entrypoint()
def main(
    tensor_parallel_size: int = 1,
    max_tokens: int = 32,
    limit: int | None = None,
) -> None:
    """组装远端命令；模型读取共享卷，结果写入个人 results volume。"""
    command = [
        "python",
        "-u",
        "-m",
        "cs336_alignment.evaluate_mmlu",
        "--model-name-or-path",
        BASE_MODEL_PATH,
        "--output-dir",
        f"{RESULTS_VOLUME_MOUNT_PATH}/mmlu_zero_shot",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-tokens",
        str(max_tokens),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    run_command.remote(command)
