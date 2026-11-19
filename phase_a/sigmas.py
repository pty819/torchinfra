#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sigma 表（移植 sdcpp denoiser.hpp FluxScheduler, qwen2.1 常量）"""
import math
import torch

BASE, MAX_SHIFT, MAX_SEQ, BASE_SEQ, TERMINAL = 0.5, 0.9, 8192, 256, 0.02


def compute_mu(seq_len):
    m = (MAX_SHIFT - BASE) / (MAX_SEQ - BASE_SEQ)
    b = BASE - m * BASE_SEQ
    return seq_len * m + b


def time_shift(mu, t):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


def flux_sigmas(steps, seq_len):
    mu = compute_mu(seq_len)
    sig = [time_shift(mu, 1.0 - i / steps) if (1.0 - i / steps) > 0 else 0.0
           for i in range(steps + 1)]
    if TERMINAL > 0 and steps > 1:
        # denoiser.hpp:794-798: 尾端锚到 TERMINAL, 头部保持 1
        f = (1.0 - sig[steps - 1]) / (1.0 - TERMINAL)
        sig = [1.0 - (1.0 - s) / f for s in sig[:-1]] + [0.0]
    return torch.tensor(sig, dtype=torch.float32)


LIGHTNING4 = [1.0, 0.75, 0.5, 0.25, 0.0]
VIGGLE6 = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25, 0.0]
