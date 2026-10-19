"""Triton paged decode attention：FlashDecoding 式两阶段实现。

tile 维度由 grid 第三维展开（kernel 内无 KV 循环），每 program 输出一个
tile 的部分 softmax 结果（max / exp-sum / 加权 V 和），再由 torch 做
log-sum-exp 合并——这是长上下文 decode 并行化的真实手法（split-K）。

勘误：开发中曾误判"地址含多个运行时乘法会被编译器算错"。经独立复现证伪
（triton 无此 bug），真实原因是全局 token 序号误作块内行号。保留两阶段
结构因其自身价值；块内寻址一律用局部行号（offs_row）。
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

    # 块内寻址用局部行号 offs_row（0..BLOCK-1）；全局 token 序号只用于 mask。
    # （曾误用全局序号寻址导致读错区段，详见模块 docstring 的勘误。）
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
    """q [B, n_heads, 1, D]（任意布局）→ out [B, n_heads, D]。

    默认走单 kernel 循环版（A/B 实测全配置快 1.5-4.3x，见
    tools/bench_paged_attn.py）；两阶段 split-K 版保留为
    paged_decode_attention_splitk 供对比与教学。
    """
    return paged_decode_attention_loop(q, view, layer_idx)


def paged_decode_attention_splitk(q: torch.Tensor, view, layer_idx: int = 0
                                  ) -> torch.Tensor:
    """FlashDecoding 两阶段版（split-K）：tile 维并行换显存往返。"""
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


@triton.jit
def _paged_decode_loop(
    Q, K_cache, V_cache, block_table, kv_lens, Out, sm_scale,
    stride_q_b, stride_q_h,
    stride_kc_blk, stride_kc_h,
    stride_vc_blk, stride_vc_h,
    stride_bt_b, stride_bt_s,
    NQ_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """单 kernel 循环版：一个 program 遍历该 (请求, 头) 的全部 KV tile。

    相比两阶段版省掉 M/L/P 的显存往返；代价是并行度只有 B×n_heads——
    batch 小时会喂不饱 GPU（两阶段的 tile 维并行正是为此存在）。
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv_h = h // NQ_PER_KV
    kv_len = tl.load(kv_lens + b)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_row = tl.arange(0, BLOCK)
    q = tl.load(Q + b * stride_q_b + h * stride_q_h
                + offs_d).to(tl.float32)              # [D]，d 维连续

    m_i = -1.0e30
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for nb in range(0, tl.cdiv(kv_len, BLOCK)):
        blk = tl.load(block_table + b * stride_bt_b
                      + nb * stride_bt_s).to(tl.int64)
        blk = tl.maximum(blk, 0)
        mask = nb * BLOCK + offs_row < kv_len
        k = tl.load(K_cache + blk * stride_kc_blk + kv_h * stride_kc_h
                    + offs_row[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float32)
        qk = tl.sum(k * q[None, :], 1) * sm_scale
        qk = tl.where(mask, qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 0))
        alpha = tl.exp(m_i - m_new)      # 首轮 exp(-1e30-…)=0，无 NaN
        p = tl.exp(qk - m_new)
        p = tl.where(mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha
        v = tl.load(V_cache + blk * stride_vc_blk + kv_h * stride_vc_h
                    + offs_row[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(v * p[:, None], 0)
        m_i = m_new

    tl.store(Out + (b * tl.num_programs(1) + h) * HEAD_DIM + offs_d,
             (acc / l_i).to(Out.dtype.element_ty))


def paged_decode_attention_tensors(q: torch.Tensor, k_full: torch.Tensor,
                                   v_full: torch.Tensor, block_table: torch.Tensor,
                                   kv_lens: torch.Tensor, layer_idx: int,
                                   block_size: int) -> torch.Tensor:
    """张量直通入口（CUDA Graph 路径用，无 view/fancy-index）。

    q [B, n_heads, 1, D]；k/v_full [n_layers, blocks, n_kv, bs, D]；
    返回 [B, n_heads, D]。
    """
    B, H, _, D = q.shape
    k_cache = k_full[layer_idx]
    v_cache = v_full[layer_idx]
    assert k_cache.stride(2) == D and k_cache.stride(3) == 1
    assert v_cache.stride(2) == D and v_cache.stride(3) == 1
    assert q.stride(3) == 1, "q 的 head_dim 维必须连续"
    out = torch.empty(B, H, D, dtype=q.dtype, device=q.device)
    _paged_decode_loop[(B, H)](
        q, k_cache, v_cache, block_table, kv_lens, out,
        D ** -0.5,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        block_table.stride(0), block_table.stride(1),
        NQ_PER_KV=H // k_full.shape[2], HEAD_DIM=D, BLOCK=block_size)
    return out


def paged_decode_attention_loop(q: torch.Tensor, view, layer_idx: int = 0
                                ) -> torch.Tensor:
    """单 kernel 循环版（与两阶段版语义一致，供对比/大 batch 路径）。"""
    return paged_decode_attention_tensors(
        q, view.k, view.v, view.block_table, view.kv_lens, layer_idx,
        view.block_size)
