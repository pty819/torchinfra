# wsplit2 vs legacy 生产内核性能对决赛
# MHA 32/32 S=7102（qdit 生产形状, legacy 基线 88TF/9.39ms）
# GQA 32/8 S=2048/8192（旧 wsplit 基线 59.6/74.2 TF）
import sys
import torch, torch.nn.functional as F
from flydsl_kernels.fp8_wrapper import quantize_tensorwise
from flydsl_kernels.flash_attn_fp8_wsplit2 import build_flash_attn_fp8_wsplit2_module as build_new
from flydsl_kernels.flash_attn_fp8_gfx1201 import build_flash_attn_fp8_module as build_legacy

D = 128
torch.manual_seed(0)


def bench(exe, NQ, NKV, S, reps=30, warm=8):
    B = 1
    S_pad = (S + 127) // 128 * 128
    pad = S_pad - S
    ic8 = torch.int8
    q = (torch.randn(B, NQ, S, D, device='cuda') * 0.3).bfloat16()
    k = (torch.randn(B, NKV, S, D, device='cuda') * 0.3).bfloat16()
    v = (torch.randn(B, NKV, S, D, device='cuda') * 0.3).bfloat16()
    q8, sq = quantize_tensorwise(q); k8, sk = quantize_tensorwise(k); v8, sv = quantize_tensorwise(v)
    del q, k, v
    if pad:
        q_p = F.pad(q8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        k_p = F.pad(k8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        v_p = F.pad(v8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
    else:
        q_p, k_p, v_p = q8, k8, v8
    o = torch.zeros(B, S_pad, NQ * D, dtype=torch.bfloat16, device='cuda')
    import math
    scl = torch.tensor([sq * sk / math.sqrt(D) * math.log2(math.e), sv],
                       dtype=torch.float32, device='cuda')
    args = (q_p.view(-1).view(ic8), k_p.view(-1).view(ic8), v_p.view(-1).view(ic8),
            o.view(-1), B, S, S_pad, scl)
    for _ in range(warm):
        exe(*args)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        exe(*args)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / reps
    tf = 4 * B * NQ * S * S * D / (ms * 1e-3) / 1e12
    del q8, k8, v8, q_p, k_p, v_p, o
    torch.cuda.empty_cache()
    return ms, tf


cases = [
    ("MHA 32/32 S=7102", 32, 32, 7102),
    ("GQA 32/8 S=2048", 32, 8, 2048),
    ("GQA 32/8 S=8192", 32, 8, 8192),
]
for name, NQ, NKV, S in cases:
    new_exe = build_new(num_heads=NQ, head_dim=D, num_kv_heads=NKV)
    ms_n, tf_n = bench(new_exe, NQ, NKV, S)
    legacy = build_legacy(num_heads=NQ, head_dim=D, num_kv_heads=NKV, causal=False,
                          waves_per_eu=1, block_m=None, block_n=None, v_layout="transposed")
    ms_l, tf_l = bench(legacy, NQ, NKV, S)
    print(f"{name}: wsplit2 {ms_n:.3f}ms {tf_n:.1f}TF | legacy {ms_l:.3f}ms {tf_l:.1f}TF | "
          f"加速 {ms_l / ms_n:.2f}x")
print("BENCH DONE")
