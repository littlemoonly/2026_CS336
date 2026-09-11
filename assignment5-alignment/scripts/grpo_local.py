"""Run GRPO seeds on local GPU pairs through the conda ysy environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6", help="Physical GPU indices.")
    parser.add_argument("--conda-env", default="ysy")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("grpo_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def run_seed_queue(
    seeds: list[str],
    gpu_pair: tuple[str, str],
    port: int,
    conda_env: str,
    conda_prefix: str,
    grpo_args: list[str],
) -> list[tuple[str, int]]:
    failures = []
    for seed in seeds:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_pair)
        libstdcxx = str(Path(conda_prefix) / "lib/libstdc++.so.6")
        env["LD_PRELOAD"] = ":".join(filter(None, (libstdcxx, env.get("LD_PRELOAD"))))
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            "-u",
            "scripts/grpo.py",
            "--seed",
            seed,
            "--policy-device",
            "cuda:0",
            "--vllm-gpu",
            "1",
            "--vllm-port",
            str(port),
            *grpo_args,
        ]
        print(f"seed={seed} GPUs={gpu_pair}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        if result.returncode:
            failures.append((seed, result.returncode))
    return failures


def main() -> None:
    args = parse_args()
    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    gpu_pairs = list(zip(gpus[::2], gpus[1::2]))[: args.max_parallel]
    if not seeds or not gpu_pairs:
        raise ValueError("At least one seed and one pair of GPUs are required.")

    conda_prefix = subprocess.check_output(
        ["conda", "run", "-n", args.conda_env, "python", "-c", "import sys; print(sys.prefix)"],
        text=True,
    ).strip()

    grpo_args = args.grpo_args
    if grpo_args[:1] == ["--"]:
        grpo_args = grpo_args[1:]
    seed_queues = [seeds[index :: len(gpu_pairs)] for index in range(len(gpu_pairs))]
    with ThreadPoolExecutor(max_workers=len(gpu_pairs)) as executor:
        futures = [
            executor.submit(
                run_seed_queue,
                seed_queue,
                gpu_pair,
                args.base_port + index,
                args.conda_env,
                conda_prefix,
                grpo_args,
            )
            for index, (seed_queue, gpu_pair) in enumerate(zip(seed_queues, gpu_pairs, strict=True))
            if seed_queue
        ]
        failures = [failure for future in futures for failure in future.result()]
    if failures:
        raise SystemExit(f"Failed runs: {failures}")


if __name__ == "__main__":
    main()
