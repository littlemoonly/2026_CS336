import argparse
import numpy as np
import timeit
import torch
from dataclasses import asdict
from pathlib import Path
import csv

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.utils import get_device
import cs336_systems.model_config as model_config


def parse_args()->dict:
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='f', choices=['f', 'fb', 'fbo'])
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--measure_steps', type=int, default=10)
    parser.add_argument('--model_config', default='s', choices=['s', 'm', 'l', 'xl', 'm10b'])
    parser.add_argument('--device', default=None)
    parser.add_argument('--csv', default=None)

    return parser.parse_args()

if __name__=='__main__':
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 512
    args = parse_args()

    # initialize model
    str2cfg = {'s': model_config.small,
               'm': model_config.medium,
               'l': model_config.large,
               'xl': model_config.xl,
               'm10b': model_config.m10b,
               }
    
    device = get_device(args.device)
    print(f"using device : {device}")
    cfg_dict = asdict(str2cfg[args.model_config])

    model = BasicsTransformerLM(vocab_size=10000, context_length=CONTEXT_LENGTH, **cfg_dict).to(device)
    optimizer = AdamW(model.parameters()) # use default configs
    dataset = np.memmap(Path(__file__).resolve().parent / 'data' / 'TinyStoriesV2-GPT4-valid.bin', np.int64, mode='r')
    input, target = get_batch(
                    dataset=dataset, 
                    batch_size=BATCH_SIZE, 
                    context_length=model.context_length, 
                    device=device)    # [B, seq_len], 对于benchmarking使用固定的batch
    model.train()

    timings = []
    train_steps = args.warmup_steps + args.measure_steps

    print('-' * 30)
    print(f'model_config: {args.model_config}, mode : {args.mode}, device : {device}')
    
    for step in range(train_steps):
        log_time = (step >= args.warmup_steps)

        if log_time:
            start = timeit.default_timer()

        if args.mode == 'f':
            logits = model(input)
            if device == 'cuda':
                torch.cuda.synchronize()

        elif args.mode == 'fb':
            logits = model(input)
            loss = cross_entropy(logits, target)
            loss.backward()
            if device == 'cuda':
                torch.cuda.synchronize()

        elif args.mode == 'fbo':
            optimizer.zero_grad()
            logits = model(input)
            loss = cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
        
        if log_time:
            end = timeit.default_timer()
            timings.append(end - start)
            print(f"    step  {step} | timing = {end - start:.3f}")

    mean_time = np.mean(timings)
    tok_per_step = BATCH_SIZE * CONTEXT_LENGTH
    tok_per_sec = tok_per_step / mean_time
    print('-' * 30)
    print(f"RESULT | {args.model_config:5s} | {args.mode:3s} | {np.mean(timings):.6f} | {np.std(timings):.6f} | {tok_per_sec:.1f}")
    print(f"Mean : {mean_time:.3f} s, std  : {np.std(timings):.3f} s, tokens / sec = {tok_per_sec:.2f}")

    if args.csv:
        file_exists = Path(args.csv).exists()
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['model_config', 'mode', 'device', 'warmup_steps', 'measure_steps',
                                'mean_sec', 'std_sec', 'tokens_per_sec'])
            writer.writerow([args.model_config, args.mode, str(device),
                            args.warmup_steps, args.measure_steps,
                            round(np.mean(timings), 6), round(np.std(timings), 6),
                            round(tok_per_sec, 1)])
