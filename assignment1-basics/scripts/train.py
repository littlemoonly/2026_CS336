"""
Training script for Transformer language model with wandb and tqdm monitoring.
"""

import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
import wandb
from cs336_basics.nn.functional import *
from cs336_basics.nn.networks import TransformerLM
from cs336_basics.optimizer.optimizer import AdamW
from utils import get_device
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def init_wandb(args):
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        print(f"Wandb initialized: {wandb.run.name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')

    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='cs336-transformer', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='tune lr:1e-2~1e-1, 5000 steps', help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')

    parser.add_argument('--resume_checkpoint_path', default=None)
    parser.add_argument('--save_checkpoint_path', default=str(PROJECT_ROOT / 'checkpoint'))
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--val_interval', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5000)

    parser.add_argument('--step_num', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batches', type=int, default=200)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--rope_theta', type=float, default=10000.0)

    parser.add_argument('--weight_decay', type=float, default=0.1)  # TO VALI
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--opt_eps', type=float, default=1e-8)
    # HYPERPARAMS : TO VALIDATE
    parser.add_argument('--warmup_iters', type=int)
    parser.add_argument('--cosine_cycle_iters', type=int, default=5000)
    parser.add_argument('--min_learning_rate', type=float, default=1e-2)   
    parser.add_argument('--max_learning_rate', type=float, default=1e-1)  

    parser.add_argument('--clip_norm_bound', type=float)
    parser.add_argument('--clip_eps', type=float, default=1e-6)

    return parser.parse_args()

def log_param_norms(model:nn.Module):
    for name, param in model.named_parameters():
        weight_norm = param.data.norm().item()    #  Frobenius norm = L2 norm
        grad_norm = param.grad.norm().item()
        wandb.log({
            f"norms/weight/{name}": weight_norm, 
            f"norms/grad/{name}": grad_norm
        })

####### Monitoring Activations #######

def attach_activation_cache(model:nn.Module):
    """
    给模型的每一层挂上 hook，并返回激活值字典和 handles 列表
    """
    activation_cache = {}
    handles = []
    def make_hook(layer_idx):
        def hook(module, input, output):
            activation_cache[layer_idx] = {
                "input": input[0].detach(),
                "output": output.detach(),
            }
        return hook

    # 给每一层注册
    for i, block in enumerate(model.layers):
        h = block.register_forward_hook(make_hook(i))
        handles.append(h)

    return activation_cache, handles

def remove_hooks(handles):
    """清理所有注册的 hooks"""
    for h in handles:
        h.remove()

def log_activation_norms(activation_cache):
    for layer_idx, values in activation_cache.items():
        wandb.log({
            f"norms/activation/layer_{layer_idx}/input" : values['input'].norm(),   #TODO
            f"norms/activation/layer_{layer_idx}/output" : values['output'].norm(),
        })

def main():
    args = parse_args()
    #  Setup device
    device = get_device()
    print(f"using device: {device}")

    init_wandb(args)
    
    # Create checkpoint directory
    os.makedirs(args.save_checkpoint_path, exist_ok=True)

    # Initialize model
    model = TransformerLM(vocab_size=args.vocab_size, 
                          context_length=args.context_length, 
                          d_model=args.d_model, 
                          num_layers=args.num_layers, 
                          num_heads=args.num_heads, 
                          d_ff=args.d_ff, 
                          rope_theta=args.rope_theta, 
                          device=device, 
                          dtype=torch.float32)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params} parameters")
    if not args.no_wandb:
        wandb.log({'model/total_parameters': total_params})

    # Initialize optimizer
    optimizer = AdamW(params=model.parameters(), 
                      lr=1e-3, 
                      weight_decay=args.weight_decay, 
                      betas=(args.beta1, args.beta2), 
                      eps=args.opt_eps)

    # Load datasets
    train_data_path = os.path.join(args.data_dir, 'TinyStoriesV2-GPT4-train.bin')
    valid_data_path = os.path.join(args.data_dir, 'TinyStoriesV2-GPT4-valid.bin')

    train = np.memmap(train_data_path, dtype=np.int64, mode='r')
    valid = np.memmap(valid_data_path, dtype=np.int64, mode='r')

    # print(f"first 64 ids in train: {train[0:64]}")
    # print(f"first 64 ids in valid: {valid[0:64]}")

    # Resume from checkpoint if specified
    iter = 0
    if args.resume_checkpoint_path is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint_path}")
        iter = load_checkpoint(args.resume_checkpoint_path, 
                            model=model, 
                            optimizer=optimizer)
        print(f"Resumed from iteration {iter}")

    model.train()   # Set the module in training mode

    # Training loop
    pbar = tqdm(
        range(iter, args.step_num), 
        initial=iter, 
        total=args.step_num, 
        desc="Training"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_losses = []

    for step in pbar:
        if step % args.log_interval == 0:
            # 只在需要 log 时注册 hook
            activation_cache, handles = attach_activation_cache(model)
            

        # get lr from this iteration
        if args.warmup_iters is None:
            args.warmup_iters = int(args.cosine_cycle_iters * 0.1)
        lr = lr_cosine_schedule(t=step, 
                                max_learning_rate=args.max_learning_rate, 
                                min_learning_rate=args.min_learning_rate,
                                T_w=args.warmup_iters, 
                                T_c=args.cosine_cycle_iters)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        inputs, targets = get_batch(train, 
                          batch_size=args.batch_size, 
                          context_length=args.context_length, 
                          device=device)    # [B, context_len]
        optimizer.zero_grad()

        logits = model(inputs)  # [B, context_len, vocab_size]
        loss = cross_entropy(inputs=logits, targets=targets)
        loss.backward()     # 反向传播，计算梯度  
        if args.clip_norm_bound:
            gradient_clipping(model.parameters(), 
                            clip_norm_bound=args.clip_norm_bound, 
                            eps=args.clip_eps)

        # 梯度计算完成，计算 norm

        
        optimizer.step()    # 优化器根据梯度，更新参数
        train_losses.append(loss.item())


        if step % args.log_interval == 0:
            # 记录 activation
            log_param_norms(model)
            log_activation_norms(activation_cache)
            remove_hooks(handles)

            avg_loss = np.mean(train_losses[0:-100] if len(train_losses) > 100 else train_losses)
            perplexity = np.exp(avg_loss)
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                "Avg_loss": f"{avg_loss:.4f}",
                                "PPL" : f"{perplexity:.2f}", 
                                "lr" : f"{lr:6f}"})

            if not args.no_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/avg_loss': avg_loss,
                    'train/perplexity': perplexity,
                    'train/learning_rate': lr,
                    'iteration': step
                })

        # Validation
        if step % args.val_interval == 0 and step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.val_batches):
                    inputs, targets = get_batch(valid, 
                            batch_size=args.batch_size, 
                            context_length=args.context_length, 
                            device=device)
                    logits = model(inputs)  # [B, context_len, vocab_size]
                    loss = cross_entropy(inputs=logits, targets=targets)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            val_ppl = np.exp(avg_val_loss)

            tqdm.write(f"Validation Loss:{avg_val_loss:.4f} | PPL: {val_ppl:.2f}")

            if not args.no_wandb:
                wandb.log({
                    'val/loss': avg_val_loss,
                    'val/perplexity': val_ppl,
                    'iteration': step
                })

        # save checkpoint
        if step % args.save_interval == 0 and step > 0:
            ckp_name = f"{args.wandb_run_name}_{timestamp}_ckp_{step}.pt"
            out_path = os.path.join(args.save_checkpoint_path, ckp_name)
            save_checkpoint(model=model, 
                            optimizer=optimizer, 
                            iteration=step,
                            out=out_path)
            tqdm.write(f"Checkpoint saved: {out_path}")


    # Save final checkpoint
    final_ckp_path = os.path.join(args.save_checkpoint_path, f"{args.wandb_run_name}_{timestamp}_ckp_final.pt")
    save_checkpoint(model=model, 
                    optimizer=optimizer, 
                    iteration=args.step_num,
                    out=final_ckp_path)

    print(f"Final checkpoint saved: {final_ckp_path}")
    print("Training completed!")
    
    # Finish wandb
    if not args.no_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()