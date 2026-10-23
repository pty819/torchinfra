"""_scaled_mm tensorwise vs rowwise 在 rerank GEMM 形状下的速度对比。"""
import sys
import time

import torch

sys.path.insert(0, ".")


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
    torch.cuda.tunable.enable(True)
    shapes = [(767, 4096, 1024), (8782, 4096, 1024), (27143, 4096, 1024),
              (27143, 2048, 1024), (27143, 6144, 1024), (27143, 1024, 3072),
              (767, 1024, 3072)]
    for M, N, K in shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
        xq = x.clamp(-448, 448).to(torch.float8_e4m3fn)
        wq = w.clamp(-448, 448).to(torch.float8_e4m3fn)
        s_t = (x.abs().amax() / 448).float().reshape(1)
        ws_t = (w.abs().amax() / 448).float().reshape(1)
        # rowwise：A 每行一个 scale，B 每列（W 每行）一个 scale；
        # 权重侧用常数向量 = tensorwise 的值（语义等价，形状满足 rowwise 约定）
        s_r = (x.abs().amax(-1, keepdim=True) / 448).float()
        ws_r = torch.full((1, N), ws_t.item(), device="cuda")

        y_t = torch._scaled_mm(xq, wq.T, scale_a=s_t, scale_b=ws_t,
                               out_dtype=torch.bfloat16)
        y_r = torch._scaled_mm(xq, wq.T, scale_a=s_r, scale_b=ws_r,
                               out_dtype=torch.bfloat16)
        fl = 2 * M * N * K
        t_t = bench(lambda: torch._scaled_mm(xq, wq.T, scale_a=s_t,
                                             scale_b=ws_t,
                                             out_dtype=torch.bfloat16))
        t_r = bench(lambda: torch._scaled_mm(xq, wq.T, scale_a=s_r,
                                             scale_b=ws_r,
                                             out_dtype=torch.bfloat16))
        diff = (y_t.float() - y_r.float()).abs().max().item()
        print(f"M={M:5d} N={N} K={K}: tensor {t_t:.3f} ms "
              f"({fl/t_t/1e9:.0f} TF) | rowwise {t_r:.3f} ms "
              f"({fl/t_r/1e9:.0f} TF) | max diff {diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
