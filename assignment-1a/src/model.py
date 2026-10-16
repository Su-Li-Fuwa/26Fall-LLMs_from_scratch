import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple
@dataclass
class ModelArgs:
    vocab_size: int
    embed_dim: int
    inter_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    n_layers: int
    rope_theta: float
