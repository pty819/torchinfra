#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""73s 稳态的 per-kernel 拆账: compiled 模型 3 步采样取平均"""
import torch, time, sys
from loader import load_fp8

DEV = "cuda"
model = load_fp8(device=DEV, mode="fp8")
model_c = torch.compile(model)
model_c.eval()

LT, HT, WT = 23, 84, 84          # 1344² 潜空间
S = LT + HT * WT
x = torch.randn(1, 64, HT, WT, device=DEV, dtype=torch.bfloat16)
ctx = torch.randn(1, LT, 4096, device=DEV, dtype=torch.bfloat16)
t = torch.tensor([800.0], device=DEV)

@torch.no_grad()
def step():
    return model_c(x, ctx, t, None, LT)

for _ in range(3):   # warm + compile (缓存热)
    step()
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(3):
        step()
    torch.cuda.synchronize()

evs = prof.key_averages()
total = sum(e.self_device_time_total for e in evs)
rows = []
for e in evs:
    if e.self_device_time_total <= 0: continue
    rows.append((e.self_device_time_total, e.count, e.key))
rows.sort(reverse=True)
print(f"3 步总 GPU 时间 {total/1000:.1f}ms → 每步 {total/3000:.1f}ms (采样40前向≈{total/3000*40/1000:.1f}s)")
print(f"{'GPU时间%':>8} {'ms/3步':>9} {'次数':>7}  kernel")

def bucket(k):
    kl = k.lower()
    if "scaled_mm" in kl or "gemm" in kl or "blas" in kl or "matmul" in kl: return "GEMM(scaled_mm/hipblaslt)"
    if "fmha" in kl or "attention" in kl or "sdpa" in kl or "mha" in kl: return "ATTENTION(sdpa)"
    if "triton" in kl: return "INDUCTOR-FUSED(triton)"
    if "elementwise" in kl or "vectorized" in kl: return "ELEMENTWISE(未融合)"
    if "norm" in kl or "rms" in kl: return "NORM"
    if "reduce" in kl or "softmax" in kl: return "REDUCE/SOFTMAX"
    if "copy" in kl or "cast" in kl or "quant" in kl: return "CAST/COPY/QUANT"
    return "OTHER"

from collections import defaultdict
agg = defaultdict(lambda: [0.0, 0])
for ms, cnt, k in rows:
    b = bucket(k)
    agg[b][0] += ms; agg[b][1] += cnt
print("\n===== 桶汇总 =====")
for b, (ms, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print(f"{ms/total*100:6.1f}%  {ms/3:8.1f}ms/步  {cnt//3:6.0f}次/步  {b}")
print("\n===== TOP20 明细 =====")
for ms, cnt, k in rows[:20]:
    print(f"{ms/total*100:6.1f}%  {ms/3:8.1f}ms/步  {cnt//3:6.0f}次/步  {k[:90]}")
