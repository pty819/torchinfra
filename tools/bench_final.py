import sys, os
import torch
from flydsl_kernels.fp8_wrapper import quantize_tensorwise
from flydsl_kernels.flash_attn_fp8_wsplit import build_flash_attn_fp8_wsplit_module as build_old
from flydsl_kernels.flash_attn_fp8_wsplit2 import build_flash_attn_fp8_wsplit2_module as build_new
D = 128
src = open("tools/bench_wsplit2.py").read()
# 复用 bench() 定义
ns = {}
exec(src.split("cases = [")[0], ns)
bench = ns["bench"]
for wpe in (None, 1, 2, 4):
    e = build_new(num_heads=32, head_dim=D, num_kv_heads=8, waves_per_eu=wpe)
    ms, tf = bench(e, 32, 8, 8192)
    print(f"wsplit2 wpe={wpe}: GQA 8K {ms:.3f}ms {tf:.1f}TF")
e = build_old(num_heads=32, head_dim=D, num_kv_heads=8)
ms, tf = bench(e, 32, 8, 8192)
print(f"old wsplit   : GQA 8K {ms:.3f}ms {tf:.1f}TF")
print("DONE")
