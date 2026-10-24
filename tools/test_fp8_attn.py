"""fp8 注意力两个 kernel 的单元对拍。

_fp8_attn_kernel:  vs SDPA-on-dequant（kernel 的精确目标）+ vs 原始 bf16 SDPA
                   （量化输入噪声在内）
_rope_qkv_fp8_kernel: 两遍式（A 遍 amax partial / B 遍量化写 fp8）vs torch
                   fp32 rope 参考；L4SC 开关、kc/vc 段修正都过一遍
"""
import os
import sys

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchinfra.nemotron_rerank import (  # noqa: E402
    FA8_STAGES, FA8_TK, FA8_TQ, FA8_WARPS, FP8_MAX, HD, NKV, NQ, ROPE_ORIG,
    ROPE_TR, ROPE_WARPS, _fp8_attn_kernel, _rope_qkv_fp8_kernel, rope_tables,
)


def tq(t):
    s = (t.float().abs().amax() / FP8_MAX).clamp(min=1e-12)
    return (t.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s


def test_attn_kernel():
    torch.manual_seed(0)
    B, S = 2, 1500   # 非 TK 整数倍 → 边界 tile 也过
    q = torch.randn(B, NQ, S, HD, dtype=torch.bfloat16, device="cuda") * 2
    k = torch.randn(B, NKV, S, HD, dtype=torch.bfloat16, device="cuda") * 2
    v = torch.randn(B, NKV, S, HD, dtype=torch.bfloat16, device="cuda")
    q8, sq = tq(q)
    k8, sk = tq(k)
    v8, sv = tq(v)
    scl = torch.stack((sq * sk * (HD ** -0.5), sv)).contiguous()
    out = torch.empty(B, S, NQ * HD, dtype=torch.bfloat16, device="cuda")
    nt = triton.cdiv(S, FA8_TQ)
    part = torch.empty(B, NQ, nt, dtype=torch.float32, device="cuda")
    _fp8_attn_kernel[(B, NQ, nt)](
        q8, k8, v8, out, part, scl, S,
        NQ_PER_KV=NQ // NKV, D=HD, TQ=FA8_TQ, TK=FA8_TK, PMAX=FP8_MAX,
        num_warps=FA8_WARPS, num_stages=FA8_STAGES)

    def sdpa(qq, kk, vv):
        r = F.scaled_dot_product_attention(qq, kk, vv, enable_gqa=True)
        return r.transpose(1, 2).reshape(B, S, NQ * HD).float()

    # 参考 1：量化输入反量化后的 SDPA —— kernel 数学（softmax fp32、P 量化）
    # 的精确目标，只剩 P→fp8 一层噪声
    qd = (q8.float() * sq).bfloat16()
    kd = (k8.float() * sk).bfloat16()
    vd = (v8.float() * sv).bfloat16()
    ref = sdpa(qd, kd, vd)
    d1 = (out.float() - ref).abs().max().item()
    # 参考 2：原始 bf16 —— d2 主要成分是 QKV 输入量化本身（合成随机数据
    # 上 QK 量化就有 ~18% max 噪声：e4m3 逐元素 6% 且与信号同速率累积，
    # 不被 √128 平均掉），门禁是"不显著超过纯输入量化基线"，绝对值由
    # E2E 把关
    ref0 = sdpa(q, k, v)
    base = (ref - ref0).abs().max().item()
    d2 = (out.float() - ref0).abs().max().item()
    scale = ref0.abs().max().item()
    print(f"attn: |Δ dequant|={d1:.4f} |Δ 原始|={d2:.4f} 输入量化基线={base:.4f} "
          f"(输出幅度 {scale:.2f}, 相对 {d1 / scale:.2%} / {d2 / scale:.2%})")
    assert d1 / scale < 0.02, "P 量化噪声超预期"
    assert d2 <= base * 1.1, "kernel 在输入量化噪声之外引入了额外偏差"
    # PART 必须覆盖 out 的 amax（bf16 舍入差一点）
    assert part.amax().item() >= out.float().abs().max().item() - 1e-2
    assert part.amax().item() <= ref.abs().max().item() * 1.1


def _rope_ref_fp32(qkv, pos, cos, sin, l4):
    BS = qkv.shape[0]
    q = qkv[:, :NQ * HD].view(BS, NQ, HD).float()
    k = qkv[:, NQ * HD:NQ * HD + NKV * HD].view(BS, NKV, HD).float()
    v = qkv[:, NQ * HD + NKV * HD:].view(BS, NKV, HD).float()
    half = HD // 2
    cs = cos[pos.long()]
    c, s = cs[:, None, :], sin[pos.long()][:, None, :]
    a, b = q[..., :half], q[..., half:]
    q = torch.cat([a * c - b * s, b * c + a * s], -1)
    if l4:
        qs = 1 + 0.1 * torch.log1p(torch.floor(pos.float() / ROPE_ORIG))
        q = q * qs[:, None, None]
    a, b = k[..., :half], k[..., half:]
    k = torch.cat([a * c - b * s, b * c + a * s], -1)
    return q, k, v


def test_rope_fp8(l4):
    torch.manual_seed(1)
    B, S = 1, 20000 if l4 else 1536   # l4 时 S>16384 触发 llama-4 scale
    BS = B * S
    cos, sin = rope_tables(32768, "cpu")
    cos, sin = cos.cuda(), sin.cuda()
    qkv = torch.randn(BS, (NQ + 2 * NKV) * HD, dtype=torch.bfloat16,
                      device="cuda")
    pos = torch.arange(S, dtype=torch.int32, device="cuda").repeat(B)
    kc, vc = 1.7, 0.35
    q8 = torch.empty(B, NQ, S, HD, dtype=torch.float8_e4m3fn, device="cuda")
    k8 = torch.empty(B, NKV, S, HD, dtype=torch.float8_e4m3fn, device="cuda")
    v8 = torch.empty(B, NKV, S, HD, dtype=torch.float8_e4m3fn, device="cuda")
    n_rb = triton.cdiv(BS, ROPE_TR)
    partq = torch.empty(n_rb, NQ, dtype=torch.float32, device="cuda")
    partk = torch.empty(n_rb, NKV, dtype=torch.float32, device="cuda")
    partv = torch.empty(n_rb, NKV, dtype=torch.float32, device="cuda")
    scl = torch.empty(3, dtype=torch.float32, device="cuda")
    args = (qkv, cos, sin, pos, q8, k8, v8, scl, partq, partk, partv,
            qkv.stride(0), cos.stride(0), BS, S, kc, vc)
    common = dict(NQ=NQ, NKV=NKV, D=HD, HALF=HD // 2, TR=ROPE_TR, L4SC=l4,
                  ORIG=float(ROPE_ORIG), BETA=0.1, num_warps=ROPE_WARPS)
    _rope_qkv_fp8_kernel[(n_rb, NQ)](*args, QUANT=False, **common)
    sq = (partq.amax() / FP8_MAX).clamp(min=1e-12)
    sk = (partk.amax() / FP8_MAX).clamp(min=1e-12)
    sv = (partv.amax() / FP8_MAX).clamp(min=1e-12)
    scl.copy_(torch.stack((sq, sk, sv)))
    _rope_qkv_fp8_kernel[(n_rb, NQ)](*args, QUANT=True, **common)

    qr, kr, vr = _rope_ref_fp32(qkv, pos, cos, sin, l4)
    kr = (kr * kc).view(B, S, NKV, HD).transpose(1, 2)
    vr = (vr * vc).view(B, S, NKV, HD).transpose(1, 2)
    qr = qr.view(B, S, NQ, HD).transpose(1, 2)
    def err(x8, s, ref):
        return ((x8.float() * s.item()) - ref).abs().max().item(), s.item()
    dq, msq = err(q8, sq, qr)
    dk, msk = err(k8, sk, kr)
    dv, msv = err(v8, sv, vr)
    # fp8 半 ulp（amax 处 binade 步长 32）以内 = 量化本身的对的
    print(f"rope fp8 L4SC={l4}: |Δq|={dq:.3f}(≤{16 * msq:.3f}) "
          f"|Δk|={dk:.3f}(≤{16 * msk:.3f}) |Δv|={dv:.3f}(≤{16 * msv:.3f})")
    assert dq <= 16 * msq + 1e-6 and dk <= 16 * msk + 1e-6 \
        and dv <= 16 * msv + 1e-6, "fp8 rope 与 fp32 参考偏差超量化界"
    # A 遍 partial 与 B 遍实际写入的 amax 一致（两遍确定性）
    assert abs(q8.float().abs().amax().item() * msq
               - partq.amax().item()) <= 16 * msq + 1e-6


if __name__ == "__main__":
    test_attn_kernel()
    test_rope_fp8(False)
    test_rope_fp8(True)
    print("\nfp8 attn 单元测试全部通过")
