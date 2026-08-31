import torch
from torch import nn
from collections.abc import Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr:float, weight_decay:float, betas:tuple, eps:float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not len(betas) == 2:
            raise ValueError(f"Invalid length of beta parameter{betas}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        
        # defaults 保存 AdamW 的默认超参数
        defaults = {
            "lr": lr,
            "betas":betas, 
            "weight_decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)  # 父类创建 self.param_groups 以及 self.state

    @torch.no_grad()
    def step(self, closure:Callable|None=None, ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if state.get("t") is None:
                    #  Lazy state initialization，创建与参数形状, dtype, device 一致的全0 m,v
                    state["t"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                grad = p.grad
                # compute adjusted lr for iteration t
                # adjust_lr = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)

                # apply weight decay
                p.mul_(1 - weight_decay * lr)

                # 原地更新 m 和 v
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                # 一阶矩：m_t = β₁m_{t-1} + (1 - β₁)g_t
                exp_avg.mul_(betas[0]).add_(grad, alpha=1.0 - betas[0])
                # 二阶矩：v_t = β₂v_{t-1} + (1 - β₂)(g_t ⊙ g_t)
                exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1.0 - betas[1])
                # state["exp_avg"] = betas[0] * state["exp_avg"] + (1-betas[0]) * grad 这会生成新张量并替换旧张量
                #state["exp_avg_sq"] = betas[1] * state["exp_avg_sq"] + (1-betas[1]) * grad**2
                
                # apply moment-adjusted weight updates
                state["t"] += 1
                t = state["t"]
                exp_avg_adjusted = exp_avg / (1 - betas[0]**t)
                exp_avg_sq_adjusted = exp_avg_sq / (1 - betas[1]**t)
                p.sub_(exp_avg_adjusted / (torch.sqrt(exp_avg_sq_adjusted) + eps), alpha=lr)

        return loss