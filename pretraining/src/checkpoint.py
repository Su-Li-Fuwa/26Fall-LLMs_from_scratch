import numpy as np
import torch
from typing import Tuple, BinaryIO, IO
import os, json
from torch import nn, optim



def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    model_state = model.state_dict()
    opt_state = optimizer.state_dict()
    checkpoint = {'model_state' : model_state,
                  'opt_state' : opt_state,
                  'iteration' : iteration}
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['opt_state'])

    return checkpoint['iteration']

def load_all(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: optim.Optimizer,
    log_p, 
) -> int:

    current_step = load_checkpoint(src, model, optimizer)
    with open(log_p, "r") as f:
        log = json.load(f)

    return current_step, log
    