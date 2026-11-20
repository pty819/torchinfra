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


@torch.no_grad()
def sample_er_sde(model, ctx, neg_ctx, latent_shape, sigmas, cfg=4.0, seed=0,
                  device="cuda", dtype=torch.bfloat16, eta=1.0, callback=None):
    """er_sde 采样器 (照抄 sdcpp denoiser.hpp:2524, 默认 eta=1.0)。
    模型语义: v 输出, denoised = x - sigma*v; 2/3阶修正 + 200点积分 + 噪声注入"""
    import math
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, *latent_shape, generator=g, device=device, dtype=torch.float32)
    B = 2 if cfg != 1.0 and neg_ctx is not None else 1
    prefix = ctx.shape[1]
    neg_prefix = neg_ctx.shape[1] if neg_ctx is not None else 0

    def clamp_s(v): return min(max(v, 1e-6), 1.0 - 1e-4)
    def lam(v): v = clamp_s(v); return v / max(1.0 - v, 1e-6)
    def alpha(v): return 1.0 - clamp_s(v)
    def scaler(v):
        v = max(v, 0.0)
        return v * (math.exp(v ** 0.3) + 10.0)

    sigs = [min(s, 1.0 - 1e-4) if s > 1.0 else s for s in sigmas]  # 同C++: 只预夹>1的
    lams = [lam(s) for s in sigs]

    old_d, old_dd = None, None
    steps = len(sigs) - 1
    for i in range(steps):
        sigma = sigs[i]
        t = torch.tensor([sigma * 1000.0], device=device)
        v = model(x.to(dtype), ctx, t, None, prefix).float()
        if B == 2:
            v_u = model(x.to(dtype), neg_ctx, t, None, neg_prefix).float()
            v = v_u + cfg * (v - v_u)
        denoised = x - sigma * v
        stage = min(3, i + 1)
        if sigs[i + 1] == 0.0:
            x = denoised
        else:
            ls, lt = lams[i], lams[i + 1]
            as_, at = alpha(sigs[i]), alpha(sigs[i + 1])
            ss, st = scaler(ls), scaler(lt)
            r_alpha = at / as_ if as_ > 0 else 0.0
            r = st / ss if ss > 0 else 0.0
            x = r_alpha * r * x + at * (1.0 - r) * denoised
            if stage >= 2 and old_d is not None:
                dt = lt - ls
                step_sz = -dt / 200.0
                s_int, s_u = 0.0, 0.0
                for pp in range(200):
                    lp = lt + pp * step_sz
                    sp = scaler(lp)
                    if sp <= 0:
                        continue
                    s_int += 1.0 / sp
                    if stage >= 3 and old_dd is not None:
                        s_u += (lp - ls) / sp
                s_int *= step_sz
                denom_d = ls - lams[i - 1]
                denoised_d = None
                if abs(denom_d) > 1e-12:
                    denoised_d = (denoised - old_d) / denom_d
                    x = x + at * (dt + s_int * st) * denoised_d
                    if stage >= 3 and old_dd is not None:
                        denom_u = (ls - lams[i - 2]) * 0.5
                        if abs(denom_u) > 1e-12:
                            s_u *= step_sz
                            denoised_u = (denoised_d - old_dd) / denom_u
                            x = x + at * (0.5 * dt * dt + s_u * st) * denoised_u
                    old_dd = denoised_d
            noise_sq = lt * lt - ls * ls * r * r
            if eta > 0 and noise_sq > 0:
                noise = torch.randn(x.shape, generator=g, device=device, dtype=torch.float32)
                x = x + noise * (at * math.sqrt(max(noise_sq, 0.0)))
        old_d = denoised
        if callback:
            callback(i, steps)
    return x.float()
