"""Qwen3 子模块手写实现（RMSNorm / RoPE / Attention / MLP / DecoderLayer）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x32).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """half-split 布局 RoPE（与 HF Qwen3 一致）；cos/sin 恒 fp32 存储。"""

    def __init__(self, head_dim: int, max_pos: int, theta: float = 1e6,
                 device: str = "cuda"):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                            dtype=torch.float32,
                                            device=device) / head_dim))
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv)                      # [max_pos, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)          # [max_pos, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def apply_qk(self, q: torch.Tensor, k: torch.Tensor, start: int):
        s = q.shape[-2]
        pos = torch.arange(start, start + s, device=q.device)
        cos = self.cos[pos].to(q.dtype)[None, None]
        sin = self.sin[pos].to(q.dtype)[None, None]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
