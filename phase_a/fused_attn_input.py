# -*- coding: utf-8 -*-
"""B3: qk-RMSNorm + rope + fp8量化 + pad + BHSD转置 融合核 (Triton, gfx1201)
替代链: norm→rope→amax→量化→转置pad (实测 ~72ms/前向) → 目标 <10ms
两遍式: pass1 只算 amax(int位序 atomic, 非负f32保序); pass2 复算+量化直写 fp8
norm/rope 数学与 dit_forward.RMSNorm/apply_rope 对齐(含 bf16 中间舍入)"""
import math, torch
import triton
import triton.language as tl

@triton.jit
def _rope_quant_pass(x_ptr, w_ptr, cos_ptr, sin_ptr, amax_ptr, scale_ptr, out_ptr,
                     S, S_pad, eps,
                     H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
                     DO_NORM: tl.constexpr, DO_ROPE: tl.constexpr, DO_WRITE: tl.constexpr):
    # v2: 整行连续 load/store (向量化), tl.split/join 做偶奇对; v1 的 stride-2 访问是 8GB/s 病根
    pid = tl.program_id(0)
    offs_s = pid * BS + tl.arange(0, BS)
    m_ld = offs_s < S
    m_st = offs_s < S_pad
    DP: tl.constexpr = D // 2
    offs_d = tl.arange(0, D)
    offs_p = tl.arange(0, DP)
    acc = 0.0
    for h in range(H):
        x = tl.load(x_ptr + offs_s[:, None] * (H * D) + h * D + offs_d[None, :],
                    mask=m_ld[:, None], other=0.0).to(tl.float32)          # [BS,D] 连续
        if DO_NORM:
            ms = tl.sum(x * x, axis=1) / D
            r = tl.rsqrt(ms + eps)[:, None]
            w = tl.load(w_ptr + offs_d).to(tl.float32)[None, :]
            x = ((x * r).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
        if DO_ROPE:
            xe, xo = tl.split(x.reshape(BS, DP, 2))
            c = tl.load(cos_ptr + offs_s[:, None] * DP + offs_p[None, :], mask=m_ld[:, None], other=0.0)
            s_ = tl.load(sin_ptr + offs_s[:, None] * DP + offs_p[None, :], mask=m_ld[:, None], other=0.0)
            xe2 = (xe * c - xo * s_).to(tl.bfloat16).to(tl.float32)
            xo2 = (xe * s_ + xo * c).to(tl.bfloat16).to(tl.float32)
            x = tl.join(xe2, xo2).reshape(BS, D)
        if DO_WRITE:
            scale = tl.load(scale_ptr)
            tl.store(out_ptr + h * (S_pad * D) + offs_s[:, None] * D + offs_d[None, :],
                     (x / scale).to(tl.float8e4nv), mask=m_st[:, None])
        else:
            acc = tl.maximum(acc, tl.max(tl.max(tl.abs(x), axis=1), axis=0))
    if not DO_WRITE:
        tl.atomic_max(amax_ptr, acc.to(tl.int32, bitcast=True))


def _alloc_like8(x, S_pad, H, D):
    return torch.empty(1, H, S_pad, D, dtype=torch.float8_e4m3fn, device=x.device)


def attn_input_fp8_impl(q, k, v, wq, wk, cos, sin):
    """q/k: [1,S,H,D] 原始(未norm未rope); v 同形; cos/sin: [S, D/2]
    返回 (q8,k8,v8: [1,H,S_pad,D] fp8, scl: [2] f32)"""
    B, S, H, D = q.shape
    assert B == 1
    S_pad = (S + 127) // 128 * 128
    DP = D // 2
    BS = 32
    grid = ((S_pad + BS - 1) // BS,)
    outs = []
    scales = []
    amax_bufs = []
    for t, w, do_norm, do_rope in ((q, wq, True, True), (k, wk, True, True), (v, v, False, False)):
        amax_i = torch.zeros(1, dtype=torch.int32, device=q.device)
        _rope_quant_pass[grid](t, w if do_norm else w, cos, sin, amax_i, amax_i, t,
                               S, S_pad, 1e-6, H=H, D=D, BS=BS,
                               DO_NORM=do_norm, DO_ROPE=do_rope, DO_WRITE=False, num_warps=4)
        amax_bufs.append(amax_i)
    sq = amax_bufs[0].view(torch.float32).clamp_min(1e-12) / 448.0
    sk = amax_bufs[1].view(torch.float32).clamp_min(1e-12) / 448.0
    sv = amax_bufs[2].view(torch.float32).clamp_min(1e-12) / 448.0
    scl = torch.stack([sq * sk / math.sqrt(D) * math.log2(math.e), sv]).float().reshape(2)
    for (t, w, do_norm, do_rope), sc in zip(((q, wq, True, True), (k, wk, True, True), (v, v, False, False)),
                                            (sq, sk, sv)):
        o8 = _alloc_like8(t, S_pad, H, D)
        _rope_quant_pass[grid](t, w if do_norm else w, cos, sin, amax_bufs[0], sc, o8,
                               S, S_pad, 1e-6, H=H, D=D, BS=BS,
                               DO_NORM=do_norm, DO_ROPE=do_rope, DO_WRITE=True, num_warps=4)
        outs.append(o8)
    return outs[0], outs[1], outs[2], scl
