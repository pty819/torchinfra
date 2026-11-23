# Bench: vendored aiter FlyDSL bf16 flash attention (gfx1201) vs SDPA flash.
# Shapes follow the Nemotron rerank prefill path: B=1, H=32, D=128.
# TF counted as 4*B*H*S^2*D FLOPs for BOTH kernels (same yardstick).

import torch
import torch.nn.functional as F

from flydsl_kernels.wrapper import flydsl_flash_attn


def bench(fn, iters=20, warmup=5):
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
    B, H, D = 1, 32, 128
    print(f"{'S':>6} | {'flydsl ms':>10} {'TF':>7} | {'sdpa ms':>9} {'TF':>7} | {'flydsl/sdpa':>11} | max|Δ|")
    print("-" * 78)
    for S in [512, 2048, 8192, 32768]:
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
        qb = q.permute(0, 2, 1, 3).contiguous()
        kb = k.permute(0, 2, 1, 3).contiguous()
        vb = v.permute(0, 2, 1, 3).contiguous()

        t_fly = bench(lambda: flydsl_flash_attn(q, k, v, causal=False))
        t_sdpa = bench(lambda: F.scaled_dot_product_attention(qb, kb, vb))
        tf = 4 * B * H * S * S * D

        o_fly = flydsl_flash_attn(q, k, v, causal=False)
        o_sdpa = F.scaled_dot_product_attention(qb, kb, vb).permute(0, 2, 1, 3)
        d = (o_fly.float() - o_sdpa.float()).abs().max().item()

        print(
            f"{S:>6} | {t_fly:>9.2f}ms {tf/t_fly/1e9:>6.1f} | {t_sdpa:>8.2f}ms {tf/t_sdpa/1e9:>6.1f} "
            f"| {t_sdpa/t_fly:>10.2f}x | {d:.4f}"
        )


if __name__ == "__main__":
    main()
