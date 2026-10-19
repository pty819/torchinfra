"""瘦 GEMV（M≤16）微观基准：hipblas (F.linear) vs Triton 定制 kernel。

decode 的 GEMM 本质是 8/16 行的 GEMV，纯带宽型（权重只读一次）。
逐形状测实测 GB/s，量化 hipblas 的欠优程度与 Triton 的可达带宽。
"""
import torch
import triton
import triton.language as tl

SHAPES = [  # (N, K) —— w: [N, K]，x: [M, K] → y = x @ w.T
    (4096, 1024, "qkv fused"),
    (1024, 2048, "o_proj"),
    (6144, 1024, "gate_up fused"),
    (1024, 3072, "down_proj"),
    (151936, 1024, "lm_head"),
]


@triton.jit
def _gemv_kernel(X, W, Y, M, N, K, KT,
                 sxm, syn, swn, swk, sym,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 SPLIT_K: tl.constexpr):
    """y[m, n] = sum_k x[m,k] * w[n,k]；SPLIT_K 个 program 沿 K 均分再原子累加。

    M 很小（≤16），x 块常驻寄存器；W 按 [BN, BK] 瓦片流式读（纯带宽型）。
    每 program 负责 K 的 KT = K//SPLIT_K 连续一段（KT 为 BK 整数倍）。
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    base = pid_k * KT
    for k in range(KT // BK):
        kk = base + k * BK + tl.arange(0, BK)
        x = tl.load(X + offs_m[:, None] * sxm + kk[None, :],
                    mask=m_mask[:, None], other=0.0)              # [BM, BK]
        w = tl.load(W + offs_n[:, None] * swn + kk[None, :] * swk,
                    mask=n_mask[:, None], other=0.0)              # [BN, BK]
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    y_ptrs = Y + offs_m[:, None] * syn + offs_n[None, :] * sym
    mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(y_ptrs, acc.to(Y.dtype.element_ty), mask=mask)


def triton_gemv(x, w, BN=128, BK=64, SPLIT_K=4, num_warps=8):
    M, K = x.shape
    N = w.shape[0]
    if K % SPLIT_K or (K // SPLIT_K) % BK:
        raise ValueError("K 切分不整")
    BM = max(16, triton.next_power_of_2(M))
    y = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(N, BN), SPLIT_K)
    _gemv_kernel[grid](x, w, y, M, N, K, K // SPLIT_K,
                       x.stride(0), y.stride(0), w.stride(0), w.stride(1),
                       y.stride(1), BM=BM, BN=BN, BK=BK, SPLIT_K=SPLIT_K,
                       num_warps=num_warps)
    return y


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    g0 = torch.cuda.Event(enable_timing=True)
    g1 = torch.cuda.Event(enable_timing=True)
    g0.record()
    for _ in range(iters):
        fn()
    g1.record()
    torch.cuda.synchronize()
    return g0.elapsed_time(g1) / iters * 1000  # us


CONFIGS = [(128, 64, 4, 8), (128, 128, 2, 8), (64, 128, 4, 4), (128, 64, 8, 8),
           (256, 64, 2, 8), (64, 64, 8, 4), (128, 128, 4, 8), (64, 64, 16, 4)]


def main():
    torch.manual_seed(0)
    for M in (1, 8, 16):
        print(f"\n==== M={M} ====")
        for N, K, name in SHAPES:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            # 多份轮换权重：工作集 ≥192MB，绕开 L2 命中（否则测的是 L2 带宽）
            copies = max(2, min(24, (192 << 20) // (N * K * 2) + 1))
            ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
                  for _ in range(copies)]
            w_bytes = N * K * 2
            ref = (x.float() @ ws[0].T.float())

            idx = [0]
            def mm():
                torch.nn.functional.linear(x, ws[idx[0] % copies])
                idx[0] += 1
            t_mm = bench(mm)

            best = (None, 1e9)
            for bn, bk, sk, nw in CONFIGS:
                try:
                    out = triton_gemv(x, ws[0], bn, bk, sk, nw)
                    err = (out.float() - ref).abs().mean().item()
                    if err > 0.15:   # bf16 舍入基线 ~0.036，原子累加略增
                        continue

                    def run():
                        triton_gemv(x, ws[idx[0] % copies], bn, bk, sk, nw)
                        idx[0] += 1
                    t = bench(run)
                    if t < best[1]:
                        best = ((bn, bk, sk, nw), t)
                except Exception:
                    pass
            del ws
            torch.cuda.empty_cache()
            cfg, t_tv = best
            gbs_mm = w_bytes / 1e9 / (t_mm / 1e6)
            if cfg is None:
                print(f"{name:12s} w={w_bytes/1e9:6.3f}GB | mm {t_mm:7.1f}us "
                      f"{gbs_mm:5.0f}GB/s | triton 全部配置失败")
                continue
            gbs_tv = w_bytes / 1e9 / (t_tv / 1e6)
            print(f"{name:12s} w={w_bytes/1e9:6.3f}GB | mm {t_mm:7.1f}us "
                  f"{gbs_mm:5.0f}GB/s | triton {t_tv:7.1f}us {gbs_tv:5.0f}GB/s"
                  f" cfg={cfg}")


if __name__ == "__main__":
    main()
