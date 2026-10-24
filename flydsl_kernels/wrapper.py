# Thin Python wrapper for the vendored aiter FlyDSL flash attention (gfx1201).
# Modeled on ROCm/aiter aiter/ops/flydsl/fmha_kernels.py (flydsl_flash_attn_func),
# trimmed to the gfx1201 path; adds block_m/block_n pass-through for tuning.

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from .flash_attn_func_gfx1201 import build_flash_attn_func_module

# Tile size baked into the kernel (BLOCK_M); seq_len is padded to a multiple.
_KERNEL_BLOCK_M = 128

# Non-causal padding safety guard, same rationale as aiter: padded K/V tokens
# produce QK^T = 0 but exp(0) = 1 leaks into the softmax denominator and
# silently scales outputs. 0.5% ~ bf16 mantissa floor + 1 bit of margin.
_MAX_NONCAUSAL_PAD_RATIO = 0.005


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn only supports bf16/f16, got {dtype!r}")


@lru_cache(maxsize=64)
def _get_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    waves_per_eu: int,
    daz: bool,
    block_m: int | None,
    block_n: int | None,
):
    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        block_n=block_n,
    )


def flydsl_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    """Run the FlyDSL Flash Attention forward on gfx1201.

    q/k/v: [batch, seq_len, num_heads, head_dim] (BSHD), bf16/f16, self-attention
    (Lq == Lk). Returns O with the same shape/dtype as q.

    Tuning knobs: waves_per_eu (occupancy hint), block_m/block_n (builder tile
    params; defaults 128/32). The non-causal path rejects pad ratios > 0.5%.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn requires CUDA/HIP tensors")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"self-attention only; q/k/v must share shape, got "
            f"{tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensor, got rank {q.dim()}")

    batch, seq_len_real, num_heads, head_dim = q.shape
    if head_dim < 64 or head_dim % 32 != 0:
        raise ValueError(
            f"kernel requires head_dim >= 64 and head_dim % 32 == 0, got {head_dim}"
        )

    dtype_str = _torch_dtype_to_str(q.dtype)

    block_m_eff = block_m if block_m is not None else _KERNEL_BLOCK_M
    seq_len_pad = ((seq_len_real + block_m_eff - 1) // block_m_eff) * block_m_eff
    n_pad = seq_len_pad - seq_len_real
    if (
        not causal
        and n_pad > 0
        and n_pad / seq_len_pad > _MAX_NONCAUSAL_PAD_RATIO
    ):
        raise ValueError(
            f"non-causal padding ratio {n_pad}/{seq_len_pad} exceeds 0.5% "
            "threshold (padded K/V leaks into softmax denominator); pad "
            "seq_len to a block multiple before calling."
        )
    if seq_len_pad != seq_len_real:
        pad = n_pad
        q_p = F.pad(q.contiguous(), (0, 0, 0, 0, 0, pad))
        k_p = F.pad(k.contiguous(), (0, 0, 0, 0, 0, pad))
        v_p = F.pad(v.contiguous(), (0, 0, 0, 0, 0, pad))
    else:
        q_p = q.contiguous()
        k_p = k.contiguous()
        v_p = v.contiguous()

    o_p = torch.empty_like(q_p)

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        exe = _get_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            block_n=block_n,
        )
        exe(
            q_p.reshape(-1),
            k_p.reshape(-1),
            v_p.reshape(-1),
            o_p.reshape(-1),
            batch,
            seq_len_pad,
            stream=launch_stream,
        )

    if seq_len_pad != seq_len_real:
        return o_p[:, :seq_len_real, :, :].contiguous()
    return o_p
