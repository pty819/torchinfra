#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU fp8 冒烟 + 每步计时（随机 ctx——数值/速度冒烟, 非画质）"""
import time, torch, sys
from loader import load_fp8
from sample import sample
from sigmas import flux_sigmas

torch.manual_seed(0)
dev = "cuda"
t0 = time.time()
model = load_fp8(device=dev, mode="fp8")
print(f"加载: {time.time()-t0:.1f}s | 显存 {torch.cuda.memory_allocated()/1e9:.2f}GB")

W = H = int(sys.argv[1]) if len(sys.argv) > 1 else 512
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
Hl = Wl = W // 8
ctx = torch.randn(1, 64, 4096, device=dev).to(torch.bfloat16) * 0.5  # 假文本嵌入

sig = flux_sigmas(steps, (Hl // 2) * (Wl // 2)).tolist()
times = []
def cb(i, n):
    torch.cuda.synchronize(); times.append(time.time())
t0 = time.time()
x = sample(model, ctx, None, (16, Hl, Wl), sig, cfg=1.0, seed=7, device=dev)
torch.cuda.synchronize()
total = time.time() - t0
per = [times[i] - (times[i-1] if i else t0) for i in range(len(times))]
print(f"{W}² {steps}步: 采样总 {total:.1f}s | 每步 {['%.2f' % p for p in per]} | 输出 finite={torch.isfinite(x).all().item()} std={x.std().item():.3f}")
