"""Triton paged decode attention：grid=(B, n_heads)，单 query 逐块遍历分页 KV。

P2 阅读笔记的技巧在此落地：block_table 显式两级映射、int64 防溢出、
online softmax 带 -inf 防 NaN。单 query 向量点积无需 tl.dot。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, K_cache, V_cache, Out, block_table, kv_lens, sm_scale,
    stride_q_b, stride_q_h, stride_q_d,
    stride_o_b, stride_o_h, stride_o_d,
    stride_kc_blk, stride_kc_h, stride_kc_s, stride_kc_d,
    stride_vc_blk, stride_vc_h, stride_vc_s, stride_vc_d,
    stride_bt_b, stride_bt_s,
    NQ_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv_h = h // NQ_PER_KV
    kv_len = tl.load(kv_lens + b)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * stride_q_b + h * stride_q_h
                + offs_d * stride_q_d).to(tl.float32)      # [D]

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for n in range(0, kv_len, BLOCK):
        blk = tl.load(block_table + b * stride_bt_b
                      + (n // BLOCK) * stride_bt_s).to(tl.int64)
        offs_s = n + tl.arange(0, BLOCK)
        mask = offs_s < kv_len
        k = tl.load(K_cache + blk * stride_kc_blk + kv_h * stride_kc_h
                    + offs_s[:, None] * stride_kc_s
                    + offs_d[None, :] * stride_kc_d,
                    mask=mask[:, None], other=0.0).to(tl.float32)   # [S, D]
        qk = tl.sum(k * q[None, :], 1) * sm_scale                   # [S]
        qk = tl.where(mask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 0))
        alpha = tl.exp(m_i - m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)  # 防 NaN（笔记技巧）
        p = tl.exp(qk - m_new)
        p = tl.where(mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha
        v = tl.load(V_cache + blk * stride_vc_blk + kv_h * stride_vc_h
                    + offs_s[:, None] * stride_vc_s
                    + offs_d[None, :] * stride_vc_d,
                    mask=mask[:, None], other=0.0).to(tl.float32)   # [S, D]
        acc += tl.sum(v * p[:, None], 0)
        m_i = m_new

    out = acc / l_i
    tl.store(Out + b * stride_o_b + h * stride_o_h + offs_d * stride_o_d,
             out.to(Out.dtype.element_ty))


def paged_decode_attention(q: torch.Tensor, view, layer_idx: int = 0
                           ) -> torch.Tensor:
    """q [B, n_heads, 1, D]（任意布局，按 stride 寻址）→ out [B, n_heads, D]。"""
    B, H, _, D = q.shape
    n_kv = view.k.shape[2]
    assert H % n_kv == 0
    k_cache = view.k[layer_idx]        # [num_blocks, n_kv, bs, hd] 连续视图
    v_cache = view.v[layer_idx]
    out = torch.empty(B, H, D, dtype=q.dtype, device=q.device)
    grid = (B, H)
    _paged_decode_kernel[grid](
        q, k_cache, v_cache, out, view.block_table, view.kv_lens,
        D ** -0.5,
        q.stride(0), q.stride(1), q.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        v_cache.stride(3),
        view.block_table.stride(0), view.block_table.stride(1),
        NQ_PER_KV=H // n_kv, HEAD_DIM=D, BLOCK=view.block_size)
    return out
