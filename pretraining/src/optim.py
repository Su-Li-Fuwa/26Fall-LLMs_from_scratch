import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional, Tuple
import math
from typing import Iterable, Optional, Callable
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self,
                params: Iterable[torch.nn.Parameter],
                lr: float,
                betas: Tuple[float, float],
                eps: float,
                weight_decay: float):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, betas, eps, weight_decay = group['lr'], group['betas'], group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)    
                if "t" not in state:
                    state["t"] = 1

                p.add_(weight_decay*p, alpha = -lr)
                state["m"] = betas[0]*state["m"] + (1-betas[0])*p.grad
                state["v"] = betas[1]*state["v"] + (1-betas[1])*torch.square(p.grad)
                lr_hat = lr*(1-betas[1]**state["t"])**0.5/(1-betas[0]**state["t"])
                state["t"] = state["t"] + 1
                p.add_(state["m"]/(torch.sqrt(state["v"])+eps), alpha = -lr_hat)


def get_lr_cosine_schedule(it: int,
                            max_learning_rate: float,
                            min_learning_rate: float,
                            warmup_iters: int,
                            cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif warmup_iters <= it and it <= cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate)*(1+math.cos(math.pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)))/2
    else:  
        return min_learning_rate

@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter],
                      max_l2_norm: float,
                      eps = 1e-8) -> None:
    l2_norm = 0 
    for p in parameters:
        if p.grad is None:  continue
        l2_norm += torch.sum(p.grad**2)
    
    l2_norm = l2_norm**0.5

    if l2_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is None:  continue
            p.grad = max_l2_norm/(l2_norm + eps)*p.grad
    return
