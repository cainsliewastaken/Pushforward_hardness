"""A minimal GPT-2-style decoder (nanoGPT layout: pre-LayerNorm blocks, fused q/k/v, 4x GELU MLP, tied output head) for testing and
benchmarking KronQLM. Parameter names follow nanoGPT, so `exclude=('lm_head',)` hands the (tied) head to AdamW."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias); self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.n_head = c.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias); self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)

    def forward(self, x): return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.n_embd, bias=c.bias); self.attn = CausalSelfAttention(c)
        self.ln_2 = nn.LayerNorm(c.n_embd, bias=c.bias); self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd); self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.h = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight                     # tied
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith('c_proj.weight'): nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.h: x = b(x)
        return self.lm_head(self.ln_f(x))
