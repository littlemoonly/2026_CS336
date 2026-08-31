from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # 获取当前参数组的学习率

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取参数 p 对应的状态
                t = state.get("t", 0)  # 读取更新次数；不存在时默认为 0
                grad = p.grad.data     # 获取损失函数对 p 的梯度

                # 原地更新参数
                p.data -= lr / math.sqrt(t + 1) * grad

                state["t"] = t + 1  # 更新该参数的迭代次数

        return loss
    
if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step