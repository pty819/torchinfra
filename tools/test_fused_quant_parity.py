"""融合量化路径 vs 旧路径 bit-exact 微验证。

rmsnorm_amax / silu_mul_amax 吐出的 partial 与 _row_amax 全量读必须一致；
quantize_from_partial 与 quantize_tensorwise 的 (q, scale) 必须 bit-exact。
"""
import sys

import torch

sys.path.insert(0, ".")
from torchinfra.triton_kernels import triton_rmsnorm, triton_rmsnorm_amax
from torchinfra.triton_kernels.silu import silu_mul, silu_mul_amax
from torchinfra.fp8_linear import (quantize_tensorwise, quantize_from_partial,
                                   fp8_linear, fp8_linear_q)


def main():
    torch.manual_seed(0)
    ok = True
    for M, N in [(1, 1024), (767, 1024), (3000, 1024)]:
        x = (torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 2)
        w = torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.1
        h_old = triton_rmsnorm(x, w, 1e-6)
        h_new, part = triton_rmsnorm_amax(x, w, 1e-6)
        d = (h_old.float() - h_new.float()).abs().max().item()
        q_old, s_old = quantize_tensorwise(h_old.contiguous())
        q_new, s_new = quantize_from_partial(h_new.contiguous(), part)
        qd = (q_old.view(torch.uint8) != q_new.view(torch.uint8)).sum().item()
        sd = abs(s_old.item() - s_new.item()) / s_old.item()
        print(f"rmsnorm M={M}: h diff {d:.2e} | scale rel {sd:.2e} | "
              f"q mismatch {qd}", flush=True)
        ok &= d == 0.0 and qd == 0 and sd == 0.0

    for M, I in [(767, 3072), (27143, 3072)]:
        gu = (torch.randn(M, 2 * I, device="cuda", dtype=torch.bfloat16) * 3)
        a_old = silu_mul(gu)
        a_new, part = silu_mul_amax(gu)
        d = (a_old.float() - a_new.float()).abs().max().item()
        q_old, s_old = quantize_tensorwise(a_old)
        q_new, s_new = quantize_from_partial(a_new, part)
        qd = (q_old.view(torch.uint8) != q_new.view(torch.uint8)).sum().item()
        sd = abs(s_old.item() - s_new.item()) / s_old.item()
        print(f"silu_mul M={M}: act diff {d:.2e} | scale rel {sd:.2e} | "
              f"q mismatch {qd}", flush=True)
        ok &= d == 0.0 and qd == 0 and sd == 0.0

    # 端到端：fp8_linear vs fp8_linear_q
    for M, K, N in [(767, 1024, 4096), (27143, 3072, 1024)]:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
        wq, ws = (lambda t: (t[0], t[1]))(
            __import__("torchinfra.fp8_linear", fromlist=["quantize_weight"])
            .quantize_weight(w))
        y_old = fp8_linear(x, wq, ws)
        _, part = quantize_tensorwise(x.contiguous())
        # 用真实 partial 来源：手算行 amax（= _row_amax 语义）
        part2 = x.float().abs().amax(-1)
        # partial 应来自 bf16 舍入后的值——x 本身就是 bf16，一致
        y_new = fp8_linear_q(x, wq, ws, part2)
        d = (y_old.float() - y_new.float()).abs().max().item()
        print(f"fp8_linear M={M}: out diff {d:.2e}", flush=True)
        ok &= d == 0.0

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
