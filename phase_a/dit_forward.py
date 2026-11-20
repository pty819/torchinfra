#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-Image-2.1 DiT 前向（手搭，规格源: sdcpp src/model/diffusion/qwen_image_2_1.hpp）
键名与 unsloth fp8 checkpoint 一致；LoRA 键名兼容 alpacas/omni（gate_up 需拆分重映射见 lora.py）。
架构事实（Task 1 勘察 + sdcpp 源码）:
- 32 块, hidden 4096, 32 头 × head_dim 128, MLP 中间 12288, 297 分离布局 (gate_layer/proj=gate/up, out=down)
- 共享 joint attention (text 前缀 + image 后缀), QK-per-head RMSNorm, mrope axes [16,56,56] theta 1e4
- 调制: timestep-only, concat(t, 0) 双列 → 4 个 AdaLN-Zero 槽, 全部 32 块共享
  每槽: img 行=真时间步列, txt 行=零列; norm 槽乘 (w+1), gate 槽乘 tanh(w)
- img_norm1/2 与 norm_out.norm 为无参数 LayerNorm(eps 1e-6); txt_in: zRMSNorm(4096)→in_layer→gelu→out_layer
- 终层: scale = norm_out.linear(时间步列); 出 LN 后乘 (scale+1), proj_out → 64ch
"""
import math
import os as _os
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN, HEAD_DIM, N_LAYERS, INTER = 4096, 128, 32, 12288
CTX_DIM, IN_CH = 4096, 64
AXES_DIM, THETA = [16, 56, 56], 10000.0


def timestep_embedding(t, dim=256):
    """t: [B] in 0..1000 → [B, 256]。flip_sin_to_cos=True（cos 在前）。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.cos(), args.sin()], dim=-1)


_FP8_ATTN = _os.environ.get("QDIT_FP8_ATTN", "0") == "1"

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, zero_center=False):
        super().__init__()
        self.eps, self.zero_center = eps, zero_center
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight + 1 if self.zero_center else self.weight
        return (h.to(x.dtype) * w)


class FP8OrBF16Linear(nn.Module):
    """mode='bf16': 载入时反量化 (qdata*scale)，正确性基准; mode='fp8': _scaled_mm 路径。
    接口: forward(x[B,S,in]) -> [B,S,out]; 暴露 w_bf16 供 LoRA 叠加。"""

    def __init__(self, qdata, scale, mode="bf16", bias=None):
        super().__init__()
        self.mode = mode
        self.register_buffer("qdata", qdata)          # [out, in] e4m3fn
        self.register_buffer("scale", scale.reshape(-1, 1).float())  # [out,1] per-row
        self.w_bf16 = None
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        # 一次性能力探测（避免 forward 内 try/except 造成 dynamo 图断裂）
        self.fp8_ok = False
        if self.mode == "fp8" and self.qdata.is_cuda:
            try:
                n = min(32, self.qdata.shape[0], self.qdata.shape[1])
                p = torch.zeros(n, n, device=self.qdata.device).to(torch.float8_e4m3fn)
                torch._scaled_mm(p, self.qdata[:n, :n].t(),
                                 scale_a=torch.ones(n, 1, device=p.device),
                                 scale_b=torch.ones(1, n, device=p.device),
                                 out_dtype=torch.bfloat16)
                self.fp8_ok = True
            except Exception:
                self.fp8_ok = False

    @property
    def w(self):  # [out, in] bf16
        if self.w_bf16 is None:
            self.w_bf16 = (self.qdata.float() * self.scale).to(torch.bfloat16)
        return self.w_bf16

    def forward(self, x):
        if not self.fp8_ok or not x.is_cuda or x.shape[-2] < 16:
            # 小 M（如 modulation 的 M=2）不满足 _scaled_mm tile 要求; 临时反量化, 不缓存
            w = (self.qdata.float() * self.scale).to(torch.bfloat16)
            return F.linear(x, w.to(x.dtype))
        xa = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)  # [M,1] 逐 token
        xs = (xa / 448.0).float()
        xq = (x.float() / xs).clamp(-448, 448).to(torch.float8_e4m3fn)
        # RowWise: scale_a (M,1) × scale_b (1,N)
        y = torch._scaled_mm(xq.reshape(-1, xq.shape[-1]), self.qdata.t(),
                             scale_a=xs.reshape(-1, 1), scale_b=self.scale.t().contiguous(),
                             bias=self.bias, out_dtype=torch.bfloat16)
        return y.reshape(*x.shape[:-1], -1)


class Block(nn.Module):
    def __init__(self, linear_cls):
        super().__init__()
        mk = lambda i, o: linear_cls(i, o)
        self.attn_to_q = mk(HIDDEN, HIDDEN)
        self.attn_to_k = mk(HIDDEN, HIDDEN)
        self.attn_to_v = mk(HIDDEN, HIDDEN)
        self.attn_to_out = mk(HIDDEN, HIDDEN)
        self.attn_norm_q = RMSNorm(HEAD_DIM)
        self.attn_norm_k = RMSNorm(HEAD_DIM)
        self.mlp_gate = mk(HIDDEN, INTER)
        self.mlp_up = mk(HIDDEN, INTER)
        self.mlp_down = mk(INTER, HIDDEN)

    def forward(self, x, mod, pe, prefix_len):
        """x: [B,S,4096]; mod: 4 个 [B,2,4096]（列 0=img 真时间步, 列 1=txt 零）"""
        B, S, _ = x.shape
        m0i, m0t = mod[0][:, 0] + 1, mod[0][:, 1] + 1
        m1i, m1t = torch.tanh(mod[1][:, 0]), torch.tanh(mod[1][:, 1])
        m2i, m2t = mod[2][:, 0] + 1, mod[2][:, 1] + 1
        m3i, m3t = torch.tanh(mod[3][:, 0]), torch.tanh(mod[3][:, 1])

        def scale_rows(h, wi, wt):
            s = torch.cat([wt.unsqueeze(1).expand(-1, prefix_len, -1),
                           wi.unsqueeze(1).expand(-1, S - prefix_len, -1)], 1)
            return h * s.to(h.dtype)

        h = F.layer_norm(x.float(), (HIDDEN,), eps=1e-6).to(x.dtype)
        h = scale_rows(h, m0i, m0t)
        q = self.attn_norm_q(self.attn_to_q(h).view(B, S, 32, HEAD_DIM))
        k = self.attn_norm_k(self.attn_to_k(h).view(B, S, 32, HEAD_DIM))
        v = self.attn_to_v(h).view(B, S, 32, HEAD_DIM)
        q, k = apply_rope(q, pe), apply_rope(k, pe)
        if _FP8_ATTN:
            from fp8_attn import attention_fp8
            o = attention_fp8(q, k, v)                      # [B,S,H*D] bf16
        else:
            o = o.reshape(B, S, HIDDEN)
        o = self.attn_to_out(o.reshape(B, S, HIDDEN))
        x = x + scale_rows(o, m1i, m1t)

        h = F.layer_norm(x.float(), (HIDDEN,), eps=1e-6).to(x.dtype)
        h = scale_rows(h, m2i, m2t)
        h = F.silu(self.mlp_gate(h)) * self.mlp_up(h)
        x = x + scale_rows(self.mlp_down(h), m3i, m3t)
        return x


class QwenDiT(nn.Module):
    """键名映射在 load_fp8 中显式处理; 这里只做计算图。"""

    def __init__(self, num_layers=N_LAYERS, linear=nn.Linear):
        super().__init__()
        self.num_layers = num_layers
        mk = lambda i, o: linear(i, o, bias=False)
        self.txt_norm = RMSNorm(CTX_DIM, zero_center=True)   # ZeroCenter: w+1
        self.txt_in_layer = mk(CTX_DIM, HIDDEN)
        self.txt_out_layer = mk(HIDDEN, HIDDEN)
        self.img_in = mk(IN_CH, HIDDEN)
        self.t_embed_1 = nn.Linear(256, HIDDEN, bias=False)
        self.t_embed_2 = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.modulation = mk(HIDDEN, 4 * HIDDEN)
        self.norm_out_linear = mk(HIDDEN, HIDDEN)
        self.proj_out = mk(HIDDEN, IN_CH)
        self.blocks = nn.ModuleList([Block(mk) for _ in range(num_layers)])
        self._rope_cache = {}

    def _pe(self, prefix_len, h_tok, w_tok, device):
        key = (prefix_len, h_tok, w_tok)
        if key not in self._rope_cache:
            self._rope_cache[key] = build_rope(*key, device=device)
        return self._rope_cache[key]

    def forward(self, x_latent, ctx, timestep, pe, prefix_len):
        """x_latent: [B,64,Hl,Wl]; ctx: [B,Lt,4096]; timestep: [B] 0..1000; pe: rope 表 (文本段+图像段拼接顺序)"""
        B = x_latent.shape[0]
        if pe is None:  # 采样器未传则按形状缓存构建
            pe = self._pe(ctx.shape[1], x_latent.shape[2], x_latent.shape[3], x_latent.device)
        t = torch.stack([timestep, torch.zeros_like(timestep)], 1)          # [B,2]
        h = timestep_embedding(t.reshape(-1)).to(self.t_embed_1.weight.dtype)  # [2B,256]
        h = F.silu(self.t_embed_1(h))
        h = F.silu(self.t_embed_2(h))                                       # embedder 内 silu + 外部 silu
        mod_t = self.modulation(h).view(B, 2, 4, HIDDEN)                    # [B,列,4槽,4096]
        mod = [mod_t[:, :, i] for i in range(4)]                            # 每个 [B,2,4096]

        text = self.txt_out_layer(F.gelu(self.txt_in_layer(self.txt_norm(ctx)), approximate="tanh"))
        img = self.img_in(self._patchify(x_latent))                         # [B,Si,4096]
        joint = torch.cat([text, img], 1)
        for blk in self.blocks:
            joint = blk(joint, mod, pe, prefix_len)
        joint = joint[:, prefix_len:]
        scale = self.norm_out_linear(h.view(B, 2, -1)[:, 0]) + 1            # 真时间步列
        joint = F.layer_norm(joint.float(), (HIDDEN,), eps=1e-6).to(x_latent.dtype)
        joint = joint * scale.unsqueeze(1).to(joint.dtype)
        return self._unpatchify(self.proj_out(joint), x_latent.shape)

    @staticmethod
    def _patchify(x):  # [B,64,Hl,Wl] -> [B, Hl*Wl, 64]  (对齐 sdcpp patchify(x,1,1))
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)

    @staticmethod
    def _unpatchify(toks, shape):  # [B, Hl*Wl, 64] -> [B, 64, Hl, Wl]
        B, C, H, W = shape
        return toks.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------- rope (diffusers QwenEmbedRope 移植, axes [16,56,56]) ----------------

def _angles(pos, dim, theta=THETA):
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    return torch.outer(pos.float(), inv)  # [S, d/2]


def build_rope(txt_len, h_tok, w_tok, device="cpu"):
    """sdcpp QwenImage21Layout.build 移植 (axes [16,56,56]):
    text j -> (j,j,j) 从 0 起; image (h,w) -> (L_txt, h-(H-H//2), w-(W-W//2)) 中心化"""
    t_txt = torch.arange(txt_len, dtype=torch.float32)
    hh, ww = torch.meshgrid(torch.arange(h_tok), torch.arange(w_tok), indexing="ij")
    t_pos = torch.cat([t_txt, torch.full((h_tok * w_tok,), float(txt_len))])
    h_pos = torch.cat([t_txt, hh.flatten().float() - (h_tok - h_tok // 2)])
    w_pos = torch.cat([t_txt, ww.flatten().float() - (w_tok - w_tok // 2)])
    ang = torch.cat([_angles(t_pos, AXES_DIM[0]),
                     _angles(h_pos, AXES_DIM[1]),
                     _angles(w_pos, AXES_DIM[2])], dim=1)
    return ang.cos().to(device), ang.sin().to(device)


def apply_rope(x, pe):
    """x: [B,S,H,128] (interleaved pairs, 对齐 ggml rope)。pe=(cos,sin): [S,64]"""
    cos = pe[0].repeat_interleave(2, -1)[None, :, None, :]
    sin = pe[1].repeat_interleave(2, -1)[None, :, None, :]
    xr, xi = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rot = torch.stack([-xi, xr], dim=-1).flatten(3)
    return (x.float() * cos + rot * sin).to(x.dtype)
