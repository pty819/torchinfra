"""批量 prefill：同一轮装载的全部 prompt 打包成一次前向。

动机（P6 剖析）：B=32 时 32 个逐请求 eager prefill 串行 ~1.6s，占墙钟 2/3，
而 decode 稳态仅 ~3.2ms/步——prefill 是唯一还没批处理的部件。

做法（vLLM 式 packed batch）：
- 权重部分全批化：embedding/qkv/o/MLP/lm_head 都在 [sum_S, E] 上跑，
  32 个 prompt 只扫一遍权重
- attention 用 Triton varlen 因果 kernel（cu_seqlens 划界 + 在线 softmax），
  直接读本轮算出的 k/v（prefill 不需要读 cache——上下文就是自己）
- k/v 写入 paged cache 用打包 kernel（grid (B, n_kv, maxS)，块表寻址）
- lm_head 只算每个请求的最后一个位置
RoPE/QK-norm 数值与 RotaryEmbedding.apply_qk / RMSNorm 逐位同式
（cos/sin 先 cast 到 q dtype 再乘，bf16 下舍入路径一致）。
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _prefill_attn_kernel(Q, K, V, O, cu, sm_scale,
                         s_q_r, s_q_h, s_k_r, s_k_h, s_v_r, s_v_h,
                         s_o_r, s_o_h,
                         NQ_PER_KV: tl.constexpr, D: tl.constexpr,
                         TQ: tl.constexpr, TK: tl.constexpr):
    """varlen 因果注意力：Q/K/V [sum_S, H, D] 连续；cu [B+1] 划界。

    program = (请求 b, q头 h, query tile t)。kv 只扫到本 tile 末（因果），
    在线 softmax（与 decode loop kernel 同款 m/l/acc 递推）。
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    kv_h = h // NQ_PER_KV
    start = tl.load(cu + b)
    end = tl.load(cu + b + 1)
    S = end - start
    q0 = t * TQ
    if q0 >= S:
        return

    offs_q = q0 + tl.arange(0, TQ)
    offs_d = tl.arange(0, D)
    qm = offs_q < S
    # bf16 原生加载：tl.dot 输入 bf16 → tensor core，fp32 累加（P9 前
    # 曾把 q/k/v 全升 fp32 再 dot，实测 0.3 TFLOPS，占 prefill 75%）
    q = tl.load(Q + (start + offs_q[:, None]) * s_q_r + h * s_q_h
                + offs_d[None, :], mask=qm[:, None], other=0.0)

    m_i = tl.full([TQ], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([TQ], dtype=tl.float32)
    acc = tl.zeros([TQ, D], dtype=tl.float32)

    kv_end = tl.minimum(q0 + TQ, S)
    for ks in range(0, kv_end, TK):
        offs_k = ks + tl.arange(0, TK)
        km = offs_k < S
        k = tl.load(K + (start + offs_k[:, None]) * s_k_r + kv_h * s_k_h
                    + offs_d[None, :], mask=km[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        mask = qm[:, None] & km[None, :] & (offs_q[:, None] >= offs_k[None, :])
        qk = tl.where(mask, qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + (start + offs_k[:, None]) * s_v_r + kv_h * s_v_h
                    + offs_d[None, :], mask=km[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(O + (start + offs_q[:, None]) * s_o_r + h * s_o_h + offs_d[None, :],
             out.to(O.dtype.element_ty), mask=qm[:, None])


def prefill_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 cu: torch.Tensor, n_kv: int, TQ: int = 128, TK: int = 64,
                 num_warps: int = 8, num_stages: int = 2
                 ) -> torch.Tensor:
    """q [sum_S, H, D]；k/v [sum_S, n_kv, D]；cu [B+1] int32 → out [sum_S, H, D]。

    tile 与流水线参数可调：长序列时 attention 是计算瓶颈，TQ/TK/stages
    决定 tensor core 喂饱程度（见 tools/bench_prefill.py 的扫描）。"""
    S_total, H, D = q.shape
    out = torch.empty_like(q)
    max_s = int(cu.diff().max())
    _prefill_attn_kernel[(cu.shape[0] - 1, H, triton.cdiv(max_s, TQ))](
        q, k, v, out, cu, D ** -0.5,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        NQ_PER_KV=H // n_kv, D=D, TQ=TQ, TK=TK, num_warps=num_warps,
        num_stages=num_stages)
    return out


@triton.jit
def _append_prefill_kv_kernel(K_cache, V_cache, K, V, bt, cu,
                              s_k_r, s_k_h, s_v_r, s_v_h,
                              s_bt_b, s_bt_s,
                              s_kc_blk, s_kc_h, s_vc_blk, s_vc_h,
                              D: tl.constexpr, BS: tl.constexpr):
    """把打包 k/v [sum_S, n_kv, D] 按块表写入 paged cache 的 0..S_b-1 槽位。

    program = (请求 b, kv头 h, 位置 pos)；pos >= S_b 直接退出（ragged pad）。
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.program_id(2)
    start = tl.load(cu + b)
    S = tl.load(cu + b + 1) - start
    if pos >= S:
        return
    blk = tl.load(bt + b * s_bt_b + (pos // BS) * s_bt_s).to(tl.int64)
    blk = tl.maximum(blk, 0)
    offs_d = tl.arange(0, D)
    src = start + pos
    kc = K_cache + blk * s_kc_blk + h * s_kc_h + (pos % BS) * D
    tl.store(kc + offs_d, tl.load(K + src * s_k_r + h * s_k_h + offs_d))
    vc = V_cache + blk * s_vc_blk + h * s_vc_h + (pos % BS) * D
    tl.store(vc + offs_d, tl.load(V + src * s_v_r + h * s_v_h + offs_d))


@triton.jit
def _qk_rope_append_rows_kernel(
    QKV, QNW, KNW, COS, SIN, pos, bt, K_cache, V_cache, QOut, KOut,
    s_qkv_b, s_cos, s_bt_b, s_bt_s,
    s_kc_blk, s_kc_h, s_vc_blk, s_vc_h,
    eps_q, eps_k,
    NQ: tl.constexpr, NKV: tl.constexpr,
    D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    """prefill 专用：每 program 处理一行（token）的全部头。

    decode 版 grid 是 (行, 头) 每程序一个头——长序列下 (S×16) 个小程序
    纯延迟开销（S=4096 实测 12.7ms）。此版一行一程序，cos/sin 行内全部
    头复用，程序数缩 16 倍。"""
    b = tl.program_id(0)
    off_h = tl.arange(0, HALF)
    base = b * s_qkv_b
    p = tl.load(pos + b)
    cos = tl.load(COS + p * s_cos + off_h)
    sin = tl.load(SIN + p * s_cos + off_h)
    blk = tl.load(bt + b * s_bt_b + (p // BLOCK) * s_bt_s).to(tl.int64)
    blk = tl.maximum(blk, 0)
    slot = (p % BLOCK) * D
    for h in tl.static_range(NQ):
        q1 = tl.load(QKV + base + h * D + off_h).to(tl.float32)
        q2 = tl.load(QKV + base + h * D + HALF + off_h).to(tl.float32)
        rms = tl.rsqrt((tl.sum(q1 * q1) + tl.sum(q2 * q2)) / D + eps_q)
        q1 = q1 * rms * tl.load(QNW + off_h).to(tl.float32)
        q2 = q2 * rms * tl.load(QNW + HALF + off_h).to(tl.float32)
        qo = QOut + (b * NQ + h) * D
        tl.store(qo + off_h, (q1 * cos - q2 * sin).to(QOut.dtype.element_ty))
        tl.store(qo + HALF + off_h,
                 (q2 * cos + q1 * sin).to(QOut.dtype.element_ty))
        if h < NKV:
            kb = base + (NQ + h) * D
            k1 = tl.load(QKV + kb + off_h).to(tl.float32)
            k2 = tl.load(QKV + kb + HALF + off_h).to(tl.float32)
            rmsk = tl.rsqrt((tl.sum(k1 * k1) + tl.sum(k2 * k2)) / D + eps_k)
            k1 = k1 * rmsk * tl.load(KNW + off_h).to(tl.float32)
            k2 = k2 * rmsk * tl.load(KNW + HALF + off_h).to(tl.float32)
            ok1 = k1 * cos - k2 * sin
            ok2 = k2 * cos + k1 * sin
            ko = KOut + (b * NKV + h) * D
            tl.store(ko + off_h, ok1.to(KOut.dtype.element_ty))
            tl.store(ko + HALF + off_h, ok2.to(KOut.dtype.element_ty))
            kc = K_cache + blk * s_kc_blk + h * s_kc_h + slot
            tl.store(kc + off_h, ok1.to(K_cache.dtype.element_ty))
            tl.store(kc + HALF + off_h, ok2.to(K_cache.dtype.element_ty))
            vb = base + (NQ + NKV + h) * D
            vc = V_cache + blk * s_vc_blk + h * s_vc_h + slot
            tl.store(vc + off_h, tl.load(QKV + vb + off_h))
            tl.store(vc + HALF + off_h, tl.load(QKV + vb + HALF + off_h))


@triton.jit
def _qk_rope_append_tiles_kernel(
    QKV, QNW, KNW, COS, SIN, pos, bt, K_cache, V_cache, QOut, KOut,
    s_qkv, s_cos, s_bt_b, s_bt_s,
    s_kc_blk, s_kc_h, s_vc_blk, s_vc_h,
    eps_q, eps_k, n_rows,
    NQ: tl.constexpr, NKV: tl.constexpr,
    D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr,
    TR: tl.constexpr,
):
    """2D 行块版：program = (行块, 头)，[TR, HALF] tile 向量化。

    行循环版（每程序一行×16 头串行）在 S=4096 实测 7.9ms；此版按
    行块向量化，归约沿 D 轴（tl.sum(x, 1)），块表寻址逐行独立。"""
    rb = tl.program_id(0)
    h = tl.program_id(1)
    rows = rb * TR + tl.arange(0, TR)
    rm = rows < n_rows
    pos_v = tl.load(pos + rows, mask=rm, other=0)
    off_h = tl.arange(0, HALF)
    cs_ptr = pos_v[:, None] * s_cos + off_h[None, :]
    cos = tl.load(COS + cs_ptr, mask=rm[:, None], other=0.0)
    sin = tl.load(SIN + cs_ptr, mask=rm[:, None], other=0.0)
    w1 = tl.load(QNW + off_h).to(tl.float32)[None, :]
    w2 = tl.load(QNW + HALF + off_h).to(tl.float32)[None, :]

    qb = rows[:, None] * s_qkv + h * D
    q1 = tl.load(QKV + qb + off_h[None, :], mask=rm[:, None],
                 other=0.0).to(tl.float32)
    q2 = tl.load(QKV + qb + HALF + off_h[None, :], mask=rm[:, None],
                 other=0.0).to(tl.float32)
    rms = tl.rsqrt((tl.sum(q1 * q1, 1) + tl.sum(q2 * q2, 1)) / D + eps_q)
    q1 = q1 * rms[:, None] * w1
    q2 = q2 * rms[:, None] * w2
    qo = (rows[:, None] * NQ + h) * D
    tl.store(QOut + qo + off_h[None, :], (q1 * cos - q2 * sin),
             mask=rm[:, None])
    tl.store(QOut + qo + HALF + off_h[None, :], (q2 * cos + q1 * sin),
             mask=rm[:, None])

    if h < NKV:
        kw1 = tl.load(KNW + off_h).to(tl.float32)[None, :]
        kw2 = tl.load(KNW + HALF + off_h).to(tl.float32)[None, :]
        kb = rows[:, None] * s_qkv + (NQ + h) * D
        k1 = tl.load(QKV + kb + off_h[None, :], mask=rm[:, None],
                     other=0.0).to(tl.float32)
        k2 = tl.load(QKV + kb + HALF + off_h[None, :], mask=rm[:, None],
                     other=0.0).to(tl.float32)
        rmsk = tl.rsqrt((tl.sum(k1 * k1, 1) + tl.sum(k2 * k2, 1)) / D + eps_k)
        k1 = k1 * rmsk[:, None] * kw1
        k2 = k2 * rmsk[:, None] * kw2
        ok1 = k1 * cos - k2 * sin
        ok2 = k2 * cos + k1 * sin
        ko = (rows[:, None] * NKV + h) * D
        tl.store(KOut + ko + off_h[None, :], ok1, mask=rm[:, None])
        tl.store(KOut + ko + HALF + off_h[None, :], ok2, mask=rm[:, None])
        blk = tl.load(bt + rows * s_bt_b + (pos_v // BLOCK) * s_bt_s,
                      mask=rm, other=0).to(tl.int64)
        blk = tl.maximum(blk, 0)
        kc = blk[:, None] * s_kc_blk + h * s_kc_h             + (pos_v % BLOCK)[:, None] * D
        tl.store(K_cache + kc + off_h[None, :], ok1, mask=rm[:, None])
        tl.store(K_cache + kc + HALF + off_h[None, :], ok2, mask=rm[:, None])
        vb = rows[:, None] * s_qkv + (NQ + NKV + h) * D
        vc = blk[:, None] * s_vc_blk + h * s_vc_h             + (pos_v % BLOCK)[:, None] * D
        tl.store(V_cache + vc + off_h[None, :],
                 tl.load(QKV + vb + off_h[None, :], mask=rm[:, None],
                         other=0.0), mask=rm[:, None])
        tl.store(V_cache + vc + HALF + off_h[None, :],
                 tl.load(QKV + vb + HALF + off_h[None, :], mask=rm[:, None],
                         other=0.0), mask=rm[:, None])


def append_prefill_kv(paged, k: torch.Tensor, v: torch.Tensor, bt: torch.Tensor,
                      cu: torch.Tensor, layer_idx: int, max_s: int) -> None:
    kc, vc = paged.k[layer_idx], paged.v[layer_idx]
    B = cu.shape[0] - 1
    _append_prefill_kv_kernel[(B, paged.n_kv, max_s)](
        kc, vc, k, v, bt, cu,
        k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        bt.stride(0), bt.stride(1),
        kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
        D=k.shape[-1], BS=paged.block_size)


@torch.no_grad()
def prefill_batch(model, paged, prompts: list[list[int]],
                  bt_rows: list[int]) -> torch.Tensor:
    """打包 prefill：返回 [B, vocab]（每请求最后位置的 logits）。

    bt_rows：各请求在 block_table 中的行号（= rid）。调用前须已 reserve。
    """
    from torchinfra.fuse import (fuse_decode_projections, mlp_forward)
    from torchinfra.triton_kernels import triton_rmsnorm
    from torchinfra.triton_kernels.silu import silu_mul
    fuse_decode_projections(model)     # 幂等；未融合时会走 cat 回退（慢 30%+）
    device = paged.k.device
    lens = [len(p) for p in prompts]
    B = len(prompts)
    total = sum(lens)
    max_s = max(lens)
    flat = torch.tensor([t for p in prompts for t in p], dtype=torch.long,
                        device=device)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lens, device=device), 0).to(torch.int32)
    req_of_row = torch.repeat_interleave(
        torch.arange(B, device=device), torch.tensor(lens, device=device))
    pos = torch.arange(total, device=device) - cu[req_of_row]   # 每请求内位置

    cos, sin = model.rope.cos, model.rope.sin           # [max_pos, D] fp32
    bt = paged.block_table.index_select(0, torch.tensor(
        bt_rows, dtype=torch.long, device=device))
    # 行分块三合一（norm+rope+cache 写入一趟）；长序列走逐行全头版
    pos32 = pos.to(torch.int32)
    bt_rows_packed = paged.block_table.index_select(0, req_of_row)
    # 长序列时注意力换 SDPA（ROCm 融合后端，S=4096 实测 67.6 vs 自研
    # kernel 25.3 TFLOPS）；短批走自研 kernel（批内打包，SDPA 需逐请求）
    use_sdpa = max_s >= 512
    x = model.model.embed_tokens(flat)                  # [sum_S, E]
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        eps = attn.q_norm.eps
        h = triton_rmsnorm(x, layer.input_layernorm.weight, eps)
        if hasattr(attn, "w_qkv"):
            qkv = F.linear(h, attn.w_qkv)
            nq = attn.n_heads * attn.hd
            nk = attn.n_kv * attn.hd
            v = qkv[:, nq + nk:].view(total, attn.n_kv, attn.hd)
        else:
            qkv = torch.cat([attn.q_proj(h), attn.k_proj(h), attn.v_proj(h)],
                            dim=-1)
            v = qkv[:, attn.n_heads * attn.hd + attn.n_kv * attn.hd:].view(
                total, attn.n_kv, attn.hd)
        q = torch.empty(total, attn.n_heads, attn.hd, dtype=qkv.dtype,
                        device=device)
        k_out = torch.empty(total, attn.n_kv, attn.hd, dtype=qkv.dtype,
                            device=device)
        kc, vc = paged.k[i], paged.v[i]
        _qk_rope_append_tiles_kernel[(triton.cdiv(total, 16), attn.n_heads)](
            qkv, attn.q_norm.weight, attn.k_norm.weight, cos, sin,
            pos32, bt_rows_packed, kc, vc, q, k_out,
            qkv.stride(0), cos.stride(0),
            bt_rows_packed.stride(0), bt_rows_packed.stride(1),
            kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
            attn.q_norm.eps, attn.q_norm.eps, total,
            NQ=attn.n_heads, NKV=attn.n_kv, D=attn.hd, HALF=attn.hd // 2,
            BLOCK=paged.block_size, TR=16, num_warps=4)
        k = k_out

        if use_sdpa:
            outs = []
            for r in range(B):
                st, en = int(cu[r]), int(cu[r + 1])
                outs.append(F.scaled_dot_product_attention(
                    q[st:en].transpose(0, 1).unsqueeze(0),
                    k[st:en].transpose(0, 1).unsqueeze(0),
                    v[st:en].transpose(0, 1).unsqueeze(0),
                    is_causal=True, enable_gqa=True)[0].transpose(0, 1))
            out = torch.cat(outs, dim=0) if B > 1 else outs[0]
        else:
            out = prefill_attn(q, k, v.contiguous(),
                               cu, attn.n_kv)            # [sum_S, H, D]
        x = x + attn.o_proj(out.reshape(total, attn.n_heads * attn.hd))
        x = x + F.linear(
            silu_mul(F.linear(triton_rmsnorm(
                x, layer.post_attention_layernorm.weight, eps),
                layer.mlp.w_gate_up)),
            layer.mlp.down_proj.weight)
    fin = model.model.norm
    x = triton_rmsnorm(x, fin.weight, fin.eps)
    last = (cu[1:] - 1).long()                           # 每请求最后位置
    return model.lm_head(x[last])
