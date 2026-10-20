"""为什么自研 Triton 注意力输给 SDPA？三个变体拆解原因。

A masked：当前实现（每个 KV tile 都做因果掩码）
B dense ：完全无掩码（数学错，纯探针——量化掩码开销）
C split ：FA2 式循环切分（满块免掩码，仅对角 tile 掩码）
参照   ：SDPA（AMD CK 手写 HIP kernel flyc_attn_fwd）
q/k/v 布局 [H, S, D] 连续。
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _attn_var(Q, K, V, O, sm_scale,
              NQ_PER_KV: tl.constexpr, D: tl.constexpr,
              TQ: tl.constexpr, TK: tl.constexpr, MODE: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1)
    kv_h = h // NQ_PER_KV
    S = tl.num_programs(1) * TQ
    offs_q = t * TQ + tl.arange(0, TQ)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + (h * S + offs_q[:, None]) * D + offs_d[None, :])

    m_i = tl.full([TQ], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([TQ], dtype=tl.float32)
    acc = tl.zeros([TQ, D], dtype=tl.float32)
    kv_end = S if MODE == 1 else t * TQ + TQ
    for ks in range(0, kv_end, TK):
        offs_k = ks + tl.arange(0, TK)
        k_t = tl.load(K + (kv_h * S + offs_k[:, None]) * D + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k_t), out_dtype=tl.float32) * sm_scale
        need_mask = (MODE == 0) | ((MODE == 2) & (ks + TK > t * TQ))
        if need_mask:
            mask = offs_q[:, None] >= offs_k[None, :]
            qk = tl.where(mask, qk, -1.0e30)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            p = tl.where(mask, p, 0.0)
        else:
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_t = tl.load(V + (kv_h * S + offs_k[:, None]) * D + offs_d[None, :])
        acc += tl.dot(p.to(v_t.dtype), v_t, out_dtype=tl.float32)
        m_i = m_new
    tl.store(O + (h * S + offs_q[:, None]) * D + offs_d[None, :],
             (acc / l_i[:, None]).to(O.dtype.element_ty))


def bench(fn, flops, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g0 = torch.cuda.Event(enable_timing=True)
    g1 = torch.cuda.Event(enable_timing=True)
    g0.record()
    for _ in range(iters):
        fn()
    g1.record()
    torch.cuda.synchronize()
    t = g0.elapsed_time(g1) / iters
    return t, flops / (t / 1e3) / 1e12


def main():
    S, H, HKV, D = 4096, 16, 8, 128
    q = torch.randn(H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(HKV, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    fl = 2 * 2 * S * S * H * D / 2

    ref = F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True,
        enable_gqa=True)[0]
    t, tf = bench(lambda: F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True,
        enable_gqa=True), fl)
    print(f"SDPA (CK flyc)          : {t:7.2f}ms  {tf:5.1f} TFLOPS")

    for tag, mode, tq, tk, w, st in (("masked(=现实现)", 0, 128, 64, 8, 2),
                                     ("dense(无掩码探针)", 1, 128, 64, 8, 2),
                                     ("split(FA2切分)  ", 2, 128, 64, 8, 2),
                                     ("split TK=128    ", 2, 128, 128, 8, 1),
                                     ("split w=4       ", 2, 128, 64, 4, 2)):
        try:
            out = torch.empty_like(q)

            def fn():
                _attn_var[(H, S // tq)](
                    q, k, v, out, D ** -0.5,
                    NQ_PER_KV=H // HKV, D=D, TQ=tq, TK=tk, MODE=mode,
                    num_warps=w, num_stages=st)
            fn()
            err = ((out.float() - ref.float()).abs().max().item()
                   if mode != 1 else float("nan"))
            t, tf = bench(fn, fl)
            print(f"{tag} TQ={tq} TK={tk} w={w}: {t:7.2f}ms  {tf:5.1f} "
                  f"TFLOPS  err={err:.3f}")
        except Exception as e:
            print(f"{tag}: 失败 {str(e)[:80]}")


if __name__ == "__main__":
    main()
