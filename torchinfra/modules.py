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


class Attention(nn.Module):
    """GQA + QK-Norm + causal；q/k norm 在 projection 后、RoPE 前。"""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def forward(self, hidden, rope, kv, layer_idx, start_pos):
        b, s, _ = hidden.shape
        q = self.q_proj(hidden).view(b, s, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(hidden).view(b, s, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(hidden).view(b, s, self.n_kv, self.hd).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rope.apply_qk(q, k, start_pos)
        kv.append(layer_idx, k, v, start_pos)
        cur = start_pos + s
        kc, vc = kv.read(layer_idx, cur)
        kc = kc.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        vc = vc.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        scores = (q @ kc.transpose(-1, -2)) * self.scale
        if s > 1:  # causal mask；decode 单 token 不需要
            qpos = torch.arange(start_pos, cur, device=q.device)
            kpos = torch.arange(cur, device=q.device)
            scores = scores.masked_fill(
                kpos[None, None, None, :] > qpos[None, None, :, None],
                float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (probs @ vc).transpose(1, 2).reshape(b, s, self.n_heads * self.hd)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU：down(silu(gate(x)) * up(x))。"""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, rope, kv, layer_idx, start_pos):
        x = x + self.self_attn(self.input_layernorm(x), rope, kv, layer_idx, start_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
