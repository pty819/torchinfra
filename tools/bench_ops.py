"""P2 算子基准：triton.testing.do_bench，输出 markdown 表。"""
import torch
import triton
from torchinfra.modules import RMSNorm, RotaryEmbedding
from torchinfra.triton_kernels import (TritonRMSNorm, TritonRoPE,
                                       fused_rmsnorm_residual, triton_add,
                                       triton_softmax)


def bench(fn) -> float:
    return triton.testing.do_bench(fn)  # ms


def main():
    torch.manual_seed(0)
    rows = []

    # RMSNorm（引擎真实 shape：decode 时 [1, seq, 1024]）
    x = torch.randn(1, 4096, 1024, device="cuda", dtype=torch.bfloat16)
    m1 = RMSNorm(1024).cuda().to(torch.bfloat16)
    m2 = TritonRMSNorm(1024).cuda().to(torch.bfloat16)
    with torch.no_grad():
        rows.append(("RMSNorm [1,4096,1024] bf16", bench(lambda: m1(x)),
                     bench(lambda: m2(x))))

    # softmax
    s = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16) * 3
    rows.append(("softmax [32,4096] bf16",
                 bench(lambda: torch.softmax(s.float(), -1).to(torch.bfloat16)),
                 bench(lambda: triton_softmax(s))))

    # RoPE（torch 版本身含广播运算；公平对比同形状同 start）
    r1 = RotaryEmbedding(128, 8192, 1e6).cuda()
    r2 = TritonRoPE(128, 8192, 1e6).cuda()
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
    rows.append(("RoPE q16/k8×512×128 bf16",
                 bench(lambda: r1.apply_qk(q, k, 0)),
                 bench(lambda: r2.apply_qk(q, k, 0))))

    # add
    a = torch.randn(1 << 24, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    rows.append(("add 16M bf16", bench(lambda: a + b),
                 bench(lambda: triton_add(a, b))))

    # fused vs 未融合（add + rmsnorm 两步 vs 一趟）
    w = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    m3 = RMSNorm(1024).cuda().to(torch.bfloat16)
    dx = torch.randn_like(x)
    with torch.no_grad():
        def unfused():
            r = x + dx
            return m3(r)
        def fused():
            return fused_rmsnorm_residual(x, dx, w)
        rows.append(("residual+RMSNorm [1,4096,1024] bf16",
                     bench(unfused), bench(fused)))

    print("| 算子 | torch ms | triton ms | 加速比 |")
    print("|---|---|---|---|")
    for name, t_ms, tri_ms in rows:
        print(f"| {name} | {t_ms:.3f} | {tri_ms:.3f} | {t_ms / tri_ms:.2f}x |")


if __name__ == "__main__":
    main()
