# FP8 flash attention 消融矩阵：P 构造链成本 / GEMM2+V 成本 / cheap-l 收益。
# synth_p 与 no_pv 的输出是错的（诊断专用），Δ 只对 none 配置有意义。

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from flydsl_kernels.fp8_wrapper import flydsl_flash_attn_fp8, quantize_tensorwise


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.manual_seed(0)
    B, NQ, NKV, D = 1, 32, 8, 128
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    q = (torch.randn(B, NQ, S, D, device="cuda") * 0.3).bfloat16()
    k = (torch.randn(B, NKV, S, D, device="cuda") * 0.3).bfloat16()
    v = (torch.randn(B, NKV, S, D, device="cuda") * 0.3).bfloat16()
    q8, sq = quantize_tensorwise(q)
    k8, sk = quantize_tensorwise(k)
    v8, sv = quantize_tensorwise(v)
    ref = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    configs = [
        ("none", True),
        ("none", False),
        ("synth_p", True),
        ("no_pv", True),
        ("neither", True),
    ]
    times = {}
    for ablate, fl in configs:
        run = lambda a=ablate, f=fl: flydsl_flash_attn_fp8(
            q8, k8, v8, sq, sk, sv, num_kv_heads=NKV, ablate=a, faithful_l=f
        )
        o = run()
        torch.cuda.synchronize()
        if ablate == "none":
            o_bh = o.view(B, S, NQ, D).permute(0, 2, 1, 3)
            d = (o_bh.float() - ref.float()).abs().max().item()
            del o_bh
        else:
            d = float("nan")
        del o
        t = bench(run)
        times[(ablate, fl)] = t
        tf = 4 * B * NQ * S * S * D / (t * 1e-3) / 1e12
        print(f"ablate={ablate:<8} faithful_l={str(fl):<5} {t:8.2f}ms {tf:6.1f} TF  Δ={d:.4f}")

    base = times[("none", True)]
    syn = times[("synth_p", True)]
    npv = times[("no_pv", True)]
    net = times[("neither", True)]
    print(f"\n分解（总 {base:.2f}ms = 100%）:")
    print(f"  纯 GEMM1+softmax+store (neither)  : {net:6.2f}ms")
    print(f"  P 构造链 (none - synth_p)         : {base - syn:6.2f}ms")
    print(f"  GEMM2+V   (synth_p - neither)     : {syn - net:6.2f}ms")
    print(f"  exp2/fma/l 链 (no_pv - neither)   : {npv - net:6.2f}ms")
    print(f"  叠加损失 (none - 各段之和)         : {base - (net + (base - syn) + (syn - net)):6.2f}ms")
    print(f"  cheap_l 收益 (none/fl - none/fr)  : {base - times[('none', False)]:6.2f}ms")


if __name__ == "__main__":
    main()
