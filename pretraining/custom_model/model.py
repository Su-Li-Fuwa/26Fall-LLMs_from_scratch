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

# torch.set_default_device("cuda")

class KVCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = []
        self.v = []
    
    def update(self, k_n, v_n):
        self.k.append(k_n)
        self.v.append(v_n)
        
        return torch.cat(self.k, dim=1), torch.cat(self.v, dim=1)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs): 
        super().__init__()

        self.embed = Embedding(args.vocab_size, args.embed_dim)
        self.layers = torch.nn.ModuleList([Block(args, i) for i in range(args.n_layers)])
        self.n_layers = args.n_layers
        self.norm = RMSNorm(args.embed_dim)
        self.lm_head = Linear(args.embed_dim, args.vocab_size) 
        self.rotary_emb = RotaryEmbedding(args.head_dim, args.rope_theta)
        
        self.apply(self._init_weights)

        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=0.02/(2*self.n_layers)**0.5)
        
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module):
        if isinstance(module, Linear) or isinstance(module, Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, *, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor]=None,
                position_ids: Optional[torch.Tensor]=None,
                past_key_values: Optional[KVCache]=None):
        
        # B, L = attention_mask.shape
        B, L_cache = input_ids.shape

        if attention_mask is None and past_key_values is None:
            casual_mask = None
        else:
            attention_mask = input_ids.new_full([L_cache], True).unsqueeze(0).expand(B, -1)
            attention_mask = attention_mask.bool()
            K = attention_mask.shape[-1]
            q_pos = torch.arange(K-L_cache, K, device=input_ids.device)
            k_pos = torch.arange(K, device=input_ids.device)
            causal = k_pos[None, :] <= q_pos[:, None]
            casual_mask = (causal[None, :, :] & attention_mask[:, None, :])
            query_valid = attention_mask[:, -L_cache:]
            self_mask = (q_pos[:, None] == k_pos[None, :])
            casual_mask = torch.where(query_valid[:, :, None], casual_mask, self_mask[None, :, :])
        
        if position_ids is None:    position_ids = torch.arange(L_cache, device=input_ids.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(position_ids)

        block_args = {
            'attention_mask' : casual_mask,
            'position_embeddings' : position_embeddings,
            'past_key_values' : past_key_values }

        x_ = self.embed(input_ids)
        for layer in self.layers:
            x_ = layer(x_, **block_args)
        x_ = self.lm_head(self.norm(x_))
        
        return x_
    
    @torch.inference_mode()
    def generate(self, *,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                max_new_tokens: int=-1,
                temperature: float=1.0,
                top_k: int=0,
                eos_token_id: Optional[int]=None):
        
        def sampling(logits, temperature, top_k):
            if top_k > 0:
                val, idx = torch.topk(logits, top_k, dim = -1)
                logits_ = torch.full_like(logits, float('-inf'))
                logits_.scatter_(dim=-1, index=idx, src=val)
                probs = F.softmax(logits_/temperature, dim = -1)
            else:
                probs = F.softmax(logits/temperature, dim = -1)

            return torch.multinomial(probs, 1)
        
        B = input_ids.shape[0]
        valid_indices = torch.arange(B, device=input_ids.device)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        num_token = 0

        past_key_values = [[KVCache() for _ in range(B)] for l_idx in range(self.n_layers)]
        self.forward(input_ids = input_ids[valid_indices, :-1], \
                    attention_mask = attention_mask[valid_indices, :-1], \
                    position_ids = position_ids[valid_indices, :-1], \
                    past_key_values = past_key_values)  # warmup KVCache

        while len(valid_indices) > 0:
            last_logits = self.forward(input_ids=input_ids[valid_indices][:, [-1]], \
                                       attention_mask=attention_mask[valid_indices], \
                                       position_ids=position_ids[valid_indices][:, [-1]], \
                                       past_key_values = [[p[idx] for idx in valid_indices] for p in past_key_values])[:, -1, :]
            
            samples = sampling(last_logits, temperature, top_k)

            next_token = input_ids.new_full((B, 1), eos_token_id)
            next_token[valid_indices] = samples

            new_valid = []
            for idx, sample in enumerate(samples):
                if sample == eos_token_id:
                    next_token[valid_indices[idx]] = eos_token_id
                else:
                    new_valid.append(idx)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            valid_indices = valid_indices[new_valid]
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((B, 1)).bool()], dim=1)

            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
            num_token += 1
            
            if max_new_tokens != -1 and num_token >= max_new_tokens:    break

        return input_ids 
            

class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.attn = MHA(args.embed_dim, args.head_dim, args.num_attention_heads, \
                        args.num_key_value_heads, layer_idx)
        self.mlp = MLP(args.embed_dim, args.inter_dim)
        self.attn_norm = RMSNorm(args.embed_dim)
        self.mlp_norm = RMSNorm(args.embed_dim)
        self.layer_idx = layer_idx
    
    def forward(self, x: torch.Tensor, **attention_kwargs) -> torch.Tensor:
        if attention_kwargs['past_key_values'] is not None:
            pkv = attention_kwargs['past_key_values'][self.layer_idx]
        else:
            pkv = None
        attention_mask, position_embeddings, past_key_value = attention_kwargs['attention_mask'], \
                                                              attention_kwargs['position_embeddings'], \
                                                              pkv
        x_ = x + self.attn(self.attn_norm(x), attention_mask, position_embeddings, past_key_value)
        x_ = x_ + self.mlp(self.mlp_norm(x_))
        
        return x_


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty((vocab_size, dim)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((dim)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.sum(x*x, dim=-1)/self.dim + self.eps).unsqueeze(dim=-1)
        return x*self.weight*rms

class MLP(nn.Module):
    def __init__(self, embed_dim: int, inter_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.inter_dim = inter_dim
        
        self.gate_proj = Linear(embed_dim, inter_dim)
        self.up_proj = Linear(embed_dim, inter_dim)
        self.down_proj = Linear(inter_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        SiLU_gate = F.silu(gate)
        SwiGLU = SiLU_gate * up
        return self.down_proj(SwiGLU)

def scaled_dot_product_attention(
    q: torch.Tensor, # (B, H, L, head_dim)
    k: torch.Tensor, # (B, H_KV, L_KV, head_dim)
    v: torch.Tensor, # (B, H_KV, L_KV, head_dim)
    attention_mask: Optional[torch.Tensor] = None # (B, L, L_KV)
    ) -> torch.Tensor: # (B, H, L, head_dim)

    B, H, L, head_dim = q.shape
    H_KV, L_KV = k.shape[1:3]
    assert H % H_KV == 0
    
    q_ = q.reshape(B, H_KV, -1, L, head_dim)
    k_ = k.reshape(B, H_KV, 1, L_KV, head_dim)
    v_ = v.reshape(B, H_KV, 1, L_KV, head_dim)

    attn_ = q_ @ k_.transpose(-1, -2) / head_dim**0.5 # (B, H_KV, -1, L, L_KV)
    if attention_mask is not None:
        attention_mask = attention_mask.reshape(B, 1, 1, L, L_KV)
        attn_ = attn_.masked_fill(~attention_mask, float("-inf"))
    attn_ = F.softmax(attn_, dim = -1)

    attn = attn_ @ v_ # (B, H_KV, -1, L, head_dim)
    return attn.reshape(B, H, L, head_dim)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()

        self.head_dim = head_dim
        self.theta = theta 
        inv_freq = torch.Tensor([1/theta**(2*k/head_dim) for k in range(head_dim//2)])
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ptheta = position_ids.float().unsqueeze(-1)* self.inv_freq
        return torch.cos(ptheta), torch.sin(ptheta)

class MHA(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, num_attention_heads: int,
                num_key_value_heads: int, layer_idx: int): 
        super().__init__()

        self.D = embed_dim
        self.d = head_dim
        self.H = num_attention_heads
        self.H_KV = num_key_value_heads
        self.layer_idx = layer_idx

        self.q_proj = Linear(self.D, self.H*self.d)
        self.k_proj = Linear(self.D, self.H_KV*self.d)
        self.v_proj = Linear(self.D, self.H_KV*self.d)
        self.o_proj = Linear(self.H*self.d, self.D)

        self.q_norm = RMSNorm(self.d)
        self.k_norm = RMSNorm(self.d)
    
        
    def forward(self, x: torch.Tensor, # (B, L, D)
                attention_mask: torch.Tensor, # (B, L, L)
                position_embeddings: Tuple[torch.Tensor, torch.Tensor], # (B, L, d // 2)
                past_key_value: Optional[KVCache] = None) -> torch.Tensor: # (B, L, D)
        
        def apply_rope(x, cos, sin):
            """
            x:   [B, H, L, head_dim]
            cos: [B, L, head_dim // 2]
            sin: [B, L, head_dim // 2]
            """
            cos = cos.unsqueeze(1).to(dtype=x.dtype)
            sin = sin.unsqueeze(1).to(dtype=x.dtype)

            x1, x2 = x.chunk(2, dim=-1)

            return torch.cat([
                x1 * cos - x2 * sin,
                x2 * cos + x1 * sin,
            ], dim=-1)

        B, L, D = x.shape
        assert D == self.D

        q, k, v = self.q_proj(x).reshape(B, L, self.H, self.d).transpose(1,2),\
                  self.k_proj(x).reshape(B, L, self.H_KV, self.d).transpose(1,2),\
                  self.v_proj(x).reshape(B, L, self.H_KV, self.d).transpose(1,2)
        
        # (B, H, L, d)
        q, k = self.q_norm(q), self.k_norm(k)

        # q_ = torch.cat([q[:,:,:,[idx, idx+self.d//2]].unsqueeze(-2) for idx in range(self.d//2)], dim=-2)
        # k_ = torch.cat([k[:,:,:,[idx, idx+self.d//2]].unsqueeze(-2) for idx in range(self.d//2)], dim=-2)

        # Cpos, Spos = position_embeddings
        # Rot = torch.stack([torch.stack([Cpos, -Spos], dim=-1), torch.stack([Spos, Cpos], dim=-1)], dim=-2).unsqueeze(1)
        # q_rot, k_rot = Rot @ q_.unsqueeze(-1), Rot @ k_.unsqueeze(-1)

        # q = torch.cat([q_rot[:, :, :, :, 0, 0], q_rot[:, :, :, :, 1, 0]], dim=-1)
        # k = torch.cat([k_rot[:, :, :, :, 0, 0], k_rot[:, :, :, :, 1, 0]], dim=-1)

        cos, sin = position_embeddings

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_key_value is not None:
            k_new, v_new = [], []

            for batch_idx, KVCache in enumerate(past_key_value):
                k_p, v_p = KVCache.update(k[batch_idx], v[batch_idx])
                k_new.append(k_p)
                v_new.append(v_p)
            
            k = torch.stack(k_new, dim=0).to(x.device)
            v = torch.stack(v_new, dim=0).to(x.device)
        # print(q.shape, k.shape, v.shape, attention_mask.unsqueeze(1).shape)
        # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_mask.unsqueeze(1), enable_gqa=True).transpose(1,2).reshape(B, L, -1)
        #scaled_dot_product_attention(q, k, v, attention_mask).transpose(1,2).reshape(B, L, -1) # B, L, Hd

        # return self.o_proj(y) # B, L, D

        attn_mask = (
            attention_mask.unsqueeze(1)
            if attention_mask is not None
            else None
        )

        use_causal = (
            attention_mask is None
            and past_key_value is None
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=use_causal,
            dropout_p=0.0,
            enable_gqa=(self.H != self.H_KV),
        )

        y = y.transpose(1, 2).reshape(B, L, -1)

        return self.o_proj(y)

