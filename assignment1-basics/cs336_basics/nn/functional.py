import torch
from einops import rearrange, einsum
import math
import torch.nn as nn
from typing import IO, Any, BinaryIO
import os
import typing
from typing import BinaryIO, IO

def softmax(x: torch.Tensor, dimension:int=-1, temperatuer:float=1.0)->torch.Tensor:
    '''
    return: same shape as input, but its i-th dimension has a normalized distribution
    '''
    max_val, _ = torch.max(x, dim=dimension, keepdim=True)
    y = x - max_val     # broadcasting, max_val 在 dimension 维度延展到 x 的 shape
    exp = torch.exp(y / temperatuer)
    sum_exp = torch.sum(exp, dim=dimension, keepdim=True)
    return exp / sum_exp



def scaled_dot_product_attention(K, Q, V, mask:torch.Tensor | None=None):
    '''
    input:
    keys, queries of shape  (batch_size, ..., seq_len, d_k)
    values of shape         (batch_size, ..., seq_len, d_v)
    support an optional user-provided boolean mask of shape  (seq_len, seq_len)
    return an output with the shape (batch_size, ..., seq_len, d_v)
    '''
    import math
    d_k = K.shape[-1]
    Y = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = Y / math.sqrt(d_k)     # [queries, key]
    if mask is not None:
        scores.masked_fill_(~mask, -1e10)
    soft = softmax(scores, dimension=-1)
    attention = einsum(soft, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return attention


def cross_entropy(inputs:torch.Tensor, targets: torch.Tensor):
    '''
    params:
        inputs:  [..., B, vocab_size]
        targets  [..., B ] 实际上第 i 个单词的 index
    支持如下要求:
    1. Subtract the largest element for numerical stability.
    2. Cancel out log and exp whenever possible.
    3. Handle any additional batch dimensions and return the average across the batch. 
    '''
    
    # 在最后一维减去 max
    max_val, _ = torch.max(inputs, dim=-1, keepdim=True)    #  [..., B, 1]
    logits_sub = inputs - max_val   # #  [..., B, vocab]

    targets = targets.unsqueeze(-1)  # [..., B, 1]
    log_probs = logits_sub.gather(dim=-1, index=targets) - torch.log(torch.sum(torch.exp(logits_sub), dim=-1, keepdim=True))
    return torch.mean(-log_probs)

# 学习率调度
def lr_cosine_schedule(t:int, max_learning_rate:float, min_learning_rate:float, T_w:int, T_c:int):
    '''cosine learning rate schedule with warmup, returns the learning rate α'''
    if t < T_w:
        return t / T_w * max_learning_rate
    elif t <= T_c:
        return min_learning_rate + 0.5*(1 + math.cos(math.pi * (t-T_w) / (T_c-T_w))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
    
@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm:float, eps:float=1e-6)->None:
    '''
    The gradients of the parameters (parameter.grad) should be modified in-place.
    '''
    sq = 0.0
    for p in parameters:
        if p.grad is not None:
            # sum() 不传 dim 参数，默认求和所有元素，返回 0 维标量
            sq += torch.sum(p.grad ** 2).item()
    
    global_l2_norm = math.sqrt(sq)

    if global_l2_norm < max_l2_norm:
        return
    
    scale = max_l2_norm / (global_l2_norm + eps)

    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)  # in-place modify

#####################################
#######  Chap5 Training Loop  #######
#####################################

def get_batch(dataset, batch_size: int, context_length: int, device: str):
    """
    从 1D 标记数据集中随机采样一个 batch 的输入序列 x 与目标序列 y (用于自回归语言模型训练)。

    Args:
        dataset (np.ndarray): 包含 Token ID 的 1D NumPy 数组。
        batch_size (int): 批次大小。
        context_length (int): 上下文窗口长度 (Sequence Length)。
        device (str): 目标设备 ('cpu', 'cuda', 'cuda:0' 等)。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 形状均为 (batch_size, context_length) 的 LongTensor。
    """
    # import random
    # possible_start_indices = [i for i in range(len(dataset) - context_length)]
    # inputs = []
    # targets = []
    # for _ in range(batch_size):
    #     pos = random.choice(possible_start_indices)  
    #     inputs.append(torch.LongTensor(dataset[pos:pos+context_length]))
    #     targets.append(torch.LongTensor(dataset[pos+1 : pos+context_length+1]))
        
    # return (torch.stack(inputs).to(device), torch.stack(targets).to(device))
    import numpy as np
    ix = torch.randint(0, len(dataset) - context_length, (batch_size, ))

    if dataset.dtype == np.int64:
        x_list = [torch.tensor(dataset[i : i + context_length]) for i in ix]
        y_list = [torch.tensor(dataset[i + 1 : i + 1 + context_length]) for i in ix]
    else:
        x_list = [torch.tensor(dataset[i : i + context_length].astype(np.int64)) for i in ix]
        y_list = [torch.tensor(dataset[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix]

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x.to(device), y.to(device)


# Check Pointing
def save_checkpoint(model:nn.Module,
                     optimizer:torch.optim.Optimizer, 
                     iteration:int, 
                     out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {
        "model_state":model.state_dict(), 
        "optimizer_state":optimizer.state_dict(), 
        "iteration":iteration
    }
    torch.save(obj=obj, f=out)


def load_checkpoint(src:str | os.PathLike | BinaryIO | IO[bytes], 
                    model:nn.Module, 
                    optimizer:torch.optim.Optimizer
                    )->int:
    obj = torch.load(src)
    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optimizer_state"])
    return obj["iteration"]