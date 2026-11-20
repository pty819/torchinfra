#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采样循环: euler + CFG 双批。模型语义对齐 sdcpp: DiT 输出速度 v, x0 = x - sigma*v。
euler: d = (x - x0)/sigma; x += d*(sigma_next - sigma)（sdcpp denoiser.hpp:1875）"""
import torch
from sigmas import flux_sigmas, LIGHTNING4, VIGGLE6


@torch.no_grad()
def sample(model, ctx, neg_ctx, latent_shape, sigmas, cfg=1.0, seed=0,
           device="cuda", dtype=torch.bfloat16, callback=None):
    """model(x[B,64,Hl,Wl], ctx[B,L,4096], t[B], pe, prefix) → v (速度)"""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, *latent_shape, generator=g, device=device, dtype=torch.float32).to(dtype)
    B = 2 if cfg != 1.0 and neg_ctx is not None else 1
    Hl, Wl = latent_shape[1], latent_shape[2]
    n_img = Hl * Wl
    prefix = ctx.shape[1]
    neg_prefix = neg_ctx.shape[1] if neg_ctx is not None else 0
    pe = None  # 由 model 内部缓存按 (prefix,n_img) 构建；此处传 None 走缓存

    for i in range(len(sigmas) - 1):
        sigma = float(sigmas[i])
        if sigma == sigmas[i + 1]:
            continue
        t = torch.tensor([sigma * 1000.0], device=device)
        if B == 2:
            # cond/neg 各自独立前向 (不做零填充双批): 官方管线补零后靠 attention mask 屏蔽,
            # 我们无掩码实现, 拆开跑最忠实 (sdcpp 也是每序列各自真实长度)
            v_c = model(x, ctx, t, pe, prefix).float()
            v_u = model(x, neg_ctx, t, pe, neg_prefix).float()
            v = (v_u + cfg * (v_c - v_u)).to(dtype)
        else:
            v = model(x, ctx, t, pe, prefix)
        x0 = x.float() - sigma * v.float()
        d = (x.float() - x0) / sigma
        x = (x.float() + d * (sigmas[i + 1] - sigma)).to(dtype)
        if callback:
            callback(i, len(sigmas) - 1)
    return x.float()
