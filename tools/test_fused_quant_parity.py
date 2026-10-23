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
        # 同一数学、不同归约树（旧 kernel autotune 可选别的 BLOCK）：
        # 允许最后一位 ulp，但 partial 必须与"对同一 h_new 全量跑行 amax"
        # 逐位一致（max 与顺序无关），q 也随之逐位一致
        d = (h_old.float() - h_new.float()).abs().max().item()
        part_ref = h_new.view(-1, N).float().abs().amax(-1)
        pd = (part_ref - part).abs().max().item()
        q_old, s_old = quantize_tensorwise(h_new.contiguous())
        q_new, s_new = quantize_from_partial(h_new.contiguous(), part)
        qd = (q_old.view(torch.uint8) != q_new.view(torch.uint8)).sum().item()
        sd = abs(s_old.item() - s_new.item()) / s_old.item()
        print(f"rmsnorm M={M}: old/new h diff {d:.2e} | partial diff {pd:.2e} "
              f"| scale rel {sd:.2e} | q mismatch {qd}", flush=True)
        ok &= d < 1e-3 and pd == 0.0 and qd == 0 and sd == 0.0

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

    # transpose_amax：SDPA 输出 [B,H,S,D] → 转置 + 行 amax，与旧两步 bit-exact
    from torchinfra.jina_rerank import _window_attn_kernel, transpose_amax, \
        WINDOW, WIN_TQ, WIN_TK, WIN_WARPS, WIN_STAGES
    import triton
    B, H, NKV, D = 2, 16, 8, 128
    for S in [300, 27143]:
        out = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        t_ref = out.transpose(1, 2).reshape(B, S, -1).contiguous()
        t_new, part = transpose_amax(out)
        d = (t_ref.float() - t_new.float()).abs().max().item()
        pref = t_ref.view(-1, H * D).float().abs().amax(-1)
        pd = (pref - part).abs().max().item()
        print(f"transpose_amax S={S}: out diff {d:.2e}, partial diff {pd:.2e}",
              flush=True)
        ok &= d == 0.0 and pd == 0.0

    # 滑窗 kernel partial：全局 amax 与有效行的 amax 一致；左 padding tile
    # 记 0 不贡献（partial 是 tile 粒度，掩码行不写值）
    for S, padn in [(2500, 0), (2500, 300)]:  # 无 padding / 批 0 左 padding
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
        k = torch.randn(B, NKV, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
        v = torch.randn(B, NKV, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
        pad = torch.tensor([padn, 0], device="cuda", dtype=torch.int32)
        o = torch.empty(B, S, H * D, dtype=q.dtype, device=q.device)
        nt = triton.cdiv(S, WIN_TQ)
        apart = torch.empty(B, H, nt, dtype=torch.float32, device=q.device)
        _window_attn_kernel[(B, H, nt)](
            q, k, v, o, apart, pad, D ** -0.5, S,
            NQ_PER_KV=H // NKV, D=D, TQ=WIN_TQ, TK=WIN_TK, WINDOW=WINDOW,
            num_warps=WIN_WARPS, num_stages=WIN_STAGES)
        if padn == 0:
            g_ref = o.view(-1, H * D).float().abs().amax().item()
        else:  # 批 0 前 padn 行不写，核对只含有效行
            g_ref = max(o[0, padn:].float().abs().max().item(),
                        o[1].float().abs().max().item())
        g_new = apart.max().item()
        gd = abs(g_ref - g_new)
        print(f"window partial S={S} pad={padn}: global amax diff {gd:.2e}",
              flush=True)
        ok &= gd == 0.0

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
