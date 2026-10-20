"""融合 decode kernel：QK-norm + RoPE + KV cache 写入 三合一。

动机（torch.profiler 实测 B=8）：rope(×2) + q_norm + k_norm + append + rope_op
的两次 copy_ 每层共 ~7 个小 kernel / ~56 次 copyBuffer，28 层合计 ~0.5ms。
本 kernel 从融合 QKV 投影的原始输出 [B, 3E] 直接读入，一个 program 处理
一个 (请求, q头)：
  - q 分段：per-head RMSNorm → RoPE → 写 q_out [B, Hq, D]（attention 直连）
  - k 分段（h < n_kv）：per-head RMSNorm → RoPE → 写 cache 对应槽位
  - v 分段（h < n_kv）：原样直写 cache（无 norm 无 rope）
数值对齐 modules.RMSNorm（fp32 累加）与 batch_model.rope_decode_batch
（rotate-half，cos/sin fp32 表 [max_pos, D] 取前半区）。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rope_append_kernel(
    QKV, QNW, KNW, COS, SIN, starts, bt, K_cache, V_cache, QOut, KOut,
    s_qkv_b,
    s_cos,
    s_bt_b, s_bt_s,
    s_kc_blk, s_kc_h, s_vc_blk, s_vc_h,
    eps_q, eps_k,
    NQ: tl.constexpr, NKV: tl.constexpr,
    D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr,
    WRITE_KOUT: tl.constexpr,
):
    """row = program_id(0)：decode 时是请求（pos 来自 lens），prefill 复用
    时是打包 token 行（pos 传逐行位置张量，bt 传按行 gather 的块表）。
    WRITE_KOUT：额外把 rope 后的 k 写到 KOut（prefill 的 attention 直连；
    decode 不需要，constexpr 裁剪）。"""
    b = tl.program_id(0)
    h = tl.program_id(1)
    off_h = tl.arange(0, HALF)
    base = b * s_qkv_b

    q1 = tl.load(QKV + base + h * D + off_h).to(tl.float32)
    q2 = tl.load(QKV + base + h * D + HALF + off_h).to(tl.float32)
    rms_q = tl.rsqrt((tl.sum(q1 * q1) + tl.sum(q2 * q2)) / D + eps_q)
    q1 = q1 * rms_q * tl.load(QNW + off_h).to(tl.float32)
    q2 = q2 * rms_q * tl.load(QNW + HALF + off_h).to(tl.float32)

    pos = tl.load(starts + b)
    cos = tl.load(COS + pos * s_cos + off_h)
    sin = tl.load(SIN + pos * s_cos + off_h)

    qo = QOut + (b * NQ + h) * D
    tl.store(qo + off_h, (q1 * cos - q2 * sin).to(QOut.dtype.element_ty))
    tl.store(qo + HALF + off_h, (q2 * cos + q1 * sin).to(QOut.dtype.element_ty))

    if h < NKV:
        kb = base + (NQ + h) * D
        k1 = tl.load(QKV + kb + off_h).to(tl.float32)
        k2 = tl.load(QKV + kb + HALF + off_h).to(tl.float32)
        rms_k = tl.rsqrt((tl.sum(k1 * k1) + tl.sum(k2 * k2)) / D + eps_k)
        k1 = k1 * rms_k * tl.load(KNW + off_h).to(tl.float32)
        k2 = k2 * rms_k * tl.load(KNW + HALF + off_h).to(tl.float32)
        ok1 = (k1 * cos - k2 * sin)
        ok2 = (k2 * cos + k1 * sin)
        if WRITE_KOUT:
            ko = KOut + (b * NKV + h) * D
            tl.store(ko + off_h, ok1.to(KOut.dtype.element_ty))
            tl.store(ko + HALF + off_h, ok2.to(KOut.dtype.element_ty))
        blk = tl.load(bt + b * s_bt_b
                      + (pos // BLOCK) * s_bt_s).to(tl.int64)
        blk = tl.maximum(blk, 0)          # -1 槽位钳到 dummy 块 0
        slot = pos % BLOCK
        kc = K_cache + blk * s_kc_blk + h * s_kc_h + slot * D
        tl.store(kc + off_h, ok1.to(K_cache.dtype.element_ty))
        tl.store(kc + HALF + off_h, ok2.to(K_cache.dtype.element_ty))
        vb = base + (NQ + NKV + h) * D
        vc = V_cache + blk * s_vc_blk + h * s_vc_h + slot * D
        tl.store(vc + off_h, tl.load(QKV + vb + off_h))
        tl.store(vc + HALF + off_h, tl.load(QKV + vb + HALF + off_h))


def fused_qk_rope_append(qkv: torch.Tensor, qnw: torch.Tensor,
                         knw: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         starts: torch.Tensor, bt: torch.Tensor,
                         k_cache: torch.Tensor, v_cache: torch.Tensor,
                         block_size: int, layer_idx: int,
                         n_heads: int, n_kv: int, eps: float = 1e-6,
                         want_k: bool = False):
    """qkv [B, nq*D + 2*nkv*D] 连续；starts/bt 为 GPU 张量（行粒度：
    B 个行各自的位置与块表行——decode 传请求，prefill 传打包 token 行）。
    返回 rope 后的 q [B, n_heads, D]（连续）；k/v 已写入 cache。
    want_k=True 时额外返回 rope 后的 k [B, n_kv, D]（prefill attention 用）。"""
    B = qkv.shape[0]
    D = qnw.numel()
    q_out = torch.empty(B, n_heads, D, dtype=qkv.dtype, device=qkv.device)
    k_out = (torch.empty(B, n_kv, D, dtype=qkv.dtype, device=qkv.device)
             if want_k else q_out)
    kc, vc = k_cache[layer_idx], v_cache[layer_idx]
    _qk_rope_append_kernel[(B, n_heads)](
        qkv, qnw, knw, cos, sin, starts, bt, kc, vc, q_out, k_out,
        qkv.stride(0), cos.stride(0),
        bt.stride(0), bt.stride(1),
        kc.stride(0), kc.stride(1), vc.stride(0), vc.stride(1),
        eps, eps,
        NQ=n_heads, NKV=n_kv, D=D, HALF=D // 2, BLOCK=block_size,
        WRITE_KOUT=want_k)
    if want_k:
        return q_out, k_out
    return q_out
