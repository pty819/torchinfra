# Wrapper + correctness/perf test for the fp8 flash attention kernel.
# Reference: dequantize Q/K/V to fp32 (with their tensorwise scales), run
# SDPA in fp32, compare. The fp8 path's expected error is dominated by the
# e4m3 quantization of the inputs (prepared by the test itself), NOT by the
# kernel — same gate style as P14 (E2E max|Δ| ~0.03 on real weights).

import math

import torch
import torch.nn.functional as F

from flydsl_kernels.flash_attn_fp8_gfx1201 import build_flash_attn_fp8_module
from functools import lru_cache

_KERNEL_BLOCK_M = 128


@lru_cache(maxsize=64)
def _get_fp8_kernel(num_heads, head_dim, num_kv_heads, causal, waves_per_eu, block_m, block_n, v_layout, ablate, faithful_l, kv_merge, sw_pipeline):
    return build_flash_attn_fp8_module(
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        causal=causal,
        waves_per_eu=waves_per_eu,
        block_m=block_m,
        block_n=block_n,
        v_layout=v_layout,
        ablate=ablate,
        faithful_l=faithful_l,
        kv_merge=kv_merge,
        sw_pipeline=sw_pipeline,
    )


def flydsl_flash_attn_fp8(
    q8: torch.Tensor,  # [B,NQ,S,D] float8_e4m3fn (BHSD)
    k8: torch.Tensor,
    v8: torch.Tensor,
    scale_q: float | None = None,
    scale_k: float | None = None,
    scale_v: float | None = None,
    causal: bool = False,
    scl: torch.Tensor | None = None,  # device [2] f32 [qk_scale_log2e, scale_v]
    num_kv_heads: int | None = None,
    waves_per_eu: int = 1,
    stream: torch.cuda.Stream | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    v_layout: str = "transposed",
    ablate: str = "none",
    faithful_l: bool = False,  # cheap l（×448 补偿后与 faithful 同精度，+3.2%）
    kv_merge: int = 1,
    sw_pipeline: bool = False,  # staged 实验路径：单独 +1.5% 但与 cheap_l 重叠不叠加
) -> torch.Tensor:
    """fp8 flash attention: O = softmax(qk^T/sqrt(D)) v, bf16 output.

    Inputs are pre-quantized e4m3 with tensorwise scales (value = scale*int8
    repr). The kernel folds scale_q*scale_k into the softmax scale and applies
    scale_v at the epilogue.
    """
    assert q8.dtype == torch.float8_e4m3fn, f"expected fp8e4m3, got {q8.dtype}"
    B, H, S, D = q8.shape  # BHSD contiguous (GQA: k8/v8 have NKV heads)
    nkv = num_kv_heads if num_kv_heads is not None else H
    assert k8.shape == v8.shape == (B, nkv, S, D)
    if q8.stride() != (H * S * D, S * D, D, 1):
        q8 = q8.contiguous()
    if k8.stride() != (nkv * S * D, S * D, D, 1):
        k8 = k8.contiguous()
    if v8.stride() != (nkv * S * D, S * D, D, 1):
        v8 = v8.contiguous()

    block_m_eff = block_m if block_m is not None else _KERNEL_BLOCK_M
    S_pad = ((S + block_m_eff - 1) // block_m_eff) * block_m_eff
    pad = S_pad - S
    if pad > 0:
        # BHSD: pad along dim=2 (seq); padded KV rows are never read (kernel
        # bounds the kv loop by the REAL S), so any pad ratio is safe
        q_p = F.pad(q8.view(torch.int8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        k_p = F.pad(k8.view(torch.int8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        v_p = F.pad(v8.view(torch.int8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
    else:
        q_p, k_p, v_p = q8, k8, v8

    o_p = torch.empty(B, S_pad, H * D, dtype=torch.bfloat16, device=q8.device)

    with torch.cuda.device(q8.device.index):
        if scl is None:
            assert scale_q and scale_k and scale_v is not None
            qk_scale_log2e = scale_q * scale_k / math.sqrt(D) * math.log2(math.e)
            scl = torch.tensor([qk_scale_log2e, scale_v], dtype=torch.float32, device=q8.device)
        launch_stream = torch.cuda.current_stream(q8.device) if stream is None else stream
        exe = _get_fp8_kernel(H, D, nkv, causal, waves_per_eu, block_m, block_n, v_layout, ablate, faithful_l, kv_merge, sw_pipeline)
        exe(
            q_p.view(-1).view(torch.int8),
            k_p.view(-1).view(torch.int8),
            v_p.view(-1).view(torch.int8),
            o_p.view(-1),
            B,
            S,
            S_pad,
            scl,
            stream=launch_stream,
        )
    return o_p[:, :S].contiguous() if pad else o_p


def quantize_tensorwise(t: torch.Tensor):
    """torchinfra-style tensorwise e4m3 quantization; returns (q8, scale)."""
    amax = t.abs().amax().float().clamp(min=1e-12)
    scale = (amax / 448.0).item()
    q = (t.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def main():
    torch.manual_seed(0)
    # Nemotron rerank prefill shape: GQA 32 Q heads / 8 KV heads, D=128
    B, NQ, NKV, D = 1, 32, 8, 128

    for S in [2048, 8192, 32768]:
        q = (torch.randn(B, NQ, S, D, device="cuda") * 0.3).bfloat16()
        k = (torch.randn(B, NKV, S, D, device="cuda") * 0.3).bfloat16()
        v = (torch.randn(B, NKV, S, D, device="cuda") * 0.3).bfloat16()
        q8, sq = quantize_tensorwise(q)
        k8, sk = quantize_tensorwise(k)
        v8, sv = quantize_tensorwise(v)

        # reference: fp32 exact on small S; bf16 flash peer (enable_gqa) on
        # large S (fp32 falls to math backend and materializes S^2 -> OOM)
        if S <= 512:
            qf, kf, vf = (q8.float() * sq), (k8.float() * sk), (v8.float() * sv)
            ref = F.scaled_dot_product_attention(qf, kf, vf, enable_gqa=True)
            del qf, kf, vf
        else:
            ref = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        o = flydsl_flash_attn_fp8(q8, k8, v8, sq, sk, sv, num_kv_heads=NKV)
        torch.cuda.synchronize()
        o_bh = o.view(B, S, NQ, D).permute(0, 2, 1, 3)  # -> [B,NQ,S,D]
        d_ref = (o_bh.float() - ref.float()).abs().max().item()
        del ref, o_bh, o
        torch.cuda.empty_cache()

        def bench(fn, iters=20, warmup=5):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / iters

        t = bench(lambda: flydsl_flash_attn_fp8(q8, k8, v8, sq, sk, sv, num_kv_heads=NKV))
        tf = 4 * B * NQ * S * S * D / (t * 1e-3) / 1e12
        print(f"S={S:>6}: fp8(GQA) {t:8.2f}ms {tf:6.1f} TF | Δ vs dequant-ref {d_ref:.4f}")


if __name__ == "__main__":
    main()
