"""Triton paged decode attention：FlashDecoding 式两阶段实现。

背景：本机 triton 3.8.0 + gfx1201 上，"循环内计算地址 + tl.load"的模式会被
错误降级（tile≥1 时块地址算错，详见 README 踩坑记录），故 tile 维度由 grid
第三维展开（kernel 内无循环），每 program 输出一个 tile 的部分 softmax 结果，
再由 torch 做 log-sum-exp 合并。这也是真实的 FlashDecoding split-K 手法。

grid = (B, n_heads, num_tiles)；每 program 处理单个 (请求, 头, KV tile)。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_partial(
    Q, K_cache, V_cache, block_table, kv_lens, M, L, P, sm_scale,
    stride_q_b, stride_q_h,
    stride_m, stride_m_h, stride_l,
    stride_p_b, stride_p_h, stride_p_t,
    stride_kc_blk, stride_kc_h,
    stride_vc_blk, stride_vc_h,
    stride_bt_b, stride_bt_s,
    NQ_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    kv_h = h // NQ_PER_KV
    kv_len = tl.load(kv_lens + b)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_row = tl.arange(0, BLOCK)          # 块内局部行号（每 tile 自成一块）
    mask = t * BLOCK + offs_row < kv_len

    q = tl.load(Q + b * stride_q_b + h * stride_q_h
                + offs_d).to(tl.float32)                          # [D]，d 维连续
    blk = tl.load(block_table + b * stride_bt_b
                  + t * stride_bt_s).to(tl.int64)
    blk = tl.maximum(blk, 0)   # 未分配槽位是 -1：mask 全空不会解引用，夹到 0 保险

    # 本机 triton+gfx1201 陷阱：地址向量中 ≥2 个运行时 stride 乘法项相加会被
    # 错误编译。规避：运行时乘法只出现在标量 base；向量部分只乘 constexpr
    # （行 stride == HEAD_DIM 是 constexpr，d 维天然连续）。
    k_base = blk * stride_kc_blk + kv_h * stride_kc_h
    k = tl.load(K_cache + k_base
                + offs_row[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None], other=0.0).to(tl.float32)     # [S, D]
    qk = tl.sum(k * q[None, :], 1) * sm_scale                     # [S]
    qk = tl.where(mask, qk, -1.0e30)
    m_t = tl.max(qk, 0)
    p = tl.exp(qk - m_t)
    p = tl.where(mask, p, 0.0)
    l_t = tl.sum(p, 0)

    v_base = blk * stride_vc_blk + kv_h * stride_vc_h
    v = tl.load(V_cache + v_base
                + offs_row[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask[:, None], other=0.0).to(tl.float32)     # [S, D]
    acc = tl.sum(v * p[:, None], 0)                               # [D]

    tl.store(M + b * stride_m + h * stride_m_h + t, m_t)
    tl.store(L + b * stride_l + h * stride_m_h + t, l_t)
    tl.store(P + b * stride_p_b + h * stride_p_h + t * stride_p_t + offs_d, acc)


def paged_decode_attention(q: torch.Tensor, view, layer_idx: int = 0
                           ) -> torch.Tensor:
    """q [B, n_heads, 1, D]（任意布局）→ out [B, n_heads, D]。"""
    B, H, _, D = q.shape
    n_kv = view.k.shape[2]
    assert H % n_kv == 0
    k_cache = view.k[layer_idx]
    v_cache = view.v[layer_idx]
    # kernel 假定块内行 stride == HEAD_DIM、d 维连续（构造布局即如此）
    assert k_cache.stride(2) == D and k_cache.stride(3) == 1
    assert v_cache.stride(2) == D and v_cache.stride(3) == 1
    assert q.stride(3) == 1, "q 的 head_dim 维必须连续"
    max_len = int(view.kv_lens.max())
    num_tiles = triton.cdiv(max_len, view.block_size)

    m = torch.empty(B, H, num_tiles, dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    p = torch.empty(B, H, num_tiles, D, dtype=torch.float32, device=q.device)

    _paged_decode_partial[(B, H, num_tiles)](
        q, k_cache, v_cache, view.block_table, view.kv_lens, m, l, p,
        D ** -0.5,
        q.stride(0), q.stride(1),
        m.stride(0), m.stride(1), l.stride(0),
        p.stride(0), p.stride(1), p.stride(2),
        k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        view.block_table.stride(0), view.block_table.stride(1),
        NQ_PER_KV=H // n_kv, HEAD_DIM=D, BLOCK=view.block_size)

    # log-sum-exp 合并（FlashDecoding 第二阶段）
    m_g = m.max(dim=-1).values                       # [B, H]
    w = torch.exp(m - m_g[..., None])                # 空 tile → exp(-1e30 - m) = 0
    l_g = (l * w).sum(-1)                            # [B, H]
    acc = (p * w[..., None]).sum(dim=2)              # [B, H, D]
    return (acc / l_g[..., None]).to(q.dtype)
