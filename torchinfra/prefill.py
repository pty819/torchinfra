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
                 cu: torch.Tensor, n_kv: int) -> torch.Tensor:
    """q [sum_S, H, D]；k/v [sum_S, n_kv, D]；cu [B+1] int32 → out [sum_S, H, D]。"""
    S_total, H, D = q.shape
    out = torch.empty_like(q)
    max_s = int(cu.diff().max())
    TQ, TK = 64, 64
    _prefill_attn_kernel[(cu.shape[0] - 1, H, triton.cdiv(max_s, TQ))](
        q, k, v, out, cu, D ** -0.5,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), out.stride(0), out.stride(1),
        NQ_PER_KV=H // n_kv, D=D, TQ=TQ, TK=TK, num_warps=4)
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
    from torchinfra.fuse import mlp_forward
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
    # 复用 decode 的三合一 kernel：行粒度传入逐 token 位置与按行 gather
    # 的块表 → qk-norm + rope + cache 写入一趟（替代 ~18 个 eager pointwise）
    from torchinfra.fused_decode import fused_qk_rope_append
    pos32 = pos.to(torch.int32)
    bt_rows_packed = paged.block_table.index_select(0, req_of_row)
    x = model.model.embed_tokens(flat)                  # [sum_S, E]
    for i, layer in enumerate(model.model.layers):
        h = layer.input_layernorm(x)
        attn = layer.self_attn
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
        q, k = fused_qk_rope_append(
            qkv, attn.q_norm.weight, attn.k_norm.weight, cos, sin,
            pos32, bt_rows_packed, paged.k, paged.v, paged.block_size, i,
            attn.n_heads, attn.n_kv, attn.q_norm.eps, want_k=True)

        out = prefill_attn(q, k, v.contiguous(),
                           cu, attn.n_kv)                # [sum_S, H, D]
        x = x + attn.o_proj(out.reshape(total, attn.n_heads * attn.hd))
        x = x + mlp_forward(layer.post_attention_layernorm(x), layer.mlp)
    x = model.model.norm(x)
    last = (cu[1:] - 1).long()                           # 每请求最后位置
    return model.lm_head(x[last])
