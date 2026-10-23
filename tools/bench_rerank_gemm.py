"""rerank 实际 GEMM 形状：hipblas vs Triton。"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mm(X, W, Y, M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        x = tl.load(X + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(Y.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g0, g1 = torch.cuda.Event(True), torch.cuda.Event(True)
    g0.record()
    for _ in range(n):
        fn()
    g1.record()
    torch.cuda.synchronize()
    return g0.elapsed_time(g1) / n


def main():
    shapes = [(304, 4096, 1024, "qkv"), (304, 2048, 1024, "o"),
              (304, 6144, 1024, "gu"), (304, 1024, 3072, "down"),
              (1548, 4096, 1024, "qkvL"), (1548, 6144, 1024, "guL")]
    cfgs = [(64, 64, 32, 4), (128, 64, 32, 4), (64, 128, 32, 4),
            (128, 128, 32, 4), (128, 128, 64, 8), (64, 256, 32, 4)]
    for M, N, K, name in shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        t_mm = bench(lambda: torch.nn.functional.linear(x, w))
        fl = 2 * M * N * K
        best = (None, 1e9)
        for bm, bn, bk, nw in cfgs:
            y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

            def fn(bm=bm, bn=bn, bk=bk, nw=nw, y=y):
                _mm[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
                    x, w, y, M, N, K, BM=bm, BN=bn, BK=bk, num_warps=nw)
            try:
                fn()
                if (y - ref).abs().mean().item() > 0.05:
                    continue
                t = bench(fn)
                if t < best[1]:
                    best = ((bm, bn, bk, nw), t)
            except Exception:
                pass
        cfg, t = best
        print(f"M={M:4d} {name:5s} hipblas {t_mm:.3f} ms "
              f"{fl / (t_mm / 1e3) / 1e12:.0f} TF | triton {t:.3f} ms "
              f"{fl / (t / 1e3) / 1e12:.0f} TF {cfg}")


if __name__ == "__main__":
    main()
