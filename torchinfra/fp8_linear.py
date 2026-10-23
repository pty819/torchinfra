"""E4M3 tensorwise FP8 GEMM（torch._scaled_mm）。

gfx1201 上 hipBLASLt 的 tensorwise FP8 在 rerank 的全部形状（含 M≈256 的短
序列）都快过调优后的 bf16。rowwise 反而经常更慢，所以权重和激活都用一个标量
scale。激活 scale 每次现算：先按行归约 amax，再量化。
"""
import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _row_amax(x_ptr, out_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    amax = 0.0
    for off in range(0, N, BLOCK):
        mask = off + offs < N
        x = tl.load(x_ptr + row * stride + off + offs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    tl.store(out_ptr + row, amax)


@triton.jit
def _quant_e4m3(x_ptr, q_ptr, scale_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    inv = 1.0 / tl.load(scale_ptr)
    offs = tl.arange(0, BLOCK)
    for off in range(0, N, BLOCK):
        mask = off + offs < N
        x = tl.load(x_ptr + row * stride + off + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
        tl.store(q_ptr + row * N + off + offs, y.to(q_ptr.dtype.element_ty), mask=mask)


def quantize_tensorwise(x: torch.Tensor):
    """x [M, K] → (e4m3 [M, K], scale [1])。scale = amax/448。"""
    m, k = x.shape
    part = torch.empty(m, device=x.device, dtype=torch.float32)
    q = torch.empty(m, k, device=x.device, dtype=torch.float8_e4m3fn)
    block = 1024 if k <= 1024 else 2048 if k <= 2048 else 4096
    _row_amax[(m,)](x, part, x.stride(0), k, BLOCK=block, num_warps=4)
    scale = (part.amax() / FP8_MAX).clamp(min=1e-12)
    _quant_e4m3[(m,)](x, q, scale, x.stride(0), k, BLOCK=block, num_warps=4)
    return q, scale.reshape(1)


def quantize_weight(w: torch.Tensor):
    """w [N, K] → (e4m3 [N, K] 行主序, scale [1])。"""
    s = (w.detach().float().abs().amax() / FP8_MAX).clamp(min=1e-12).reshape(1)
    q = (w.detach().float() / s).to(torch.float8_e4m3fn).contiguous()
    return q, s


def fp8_linear(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor
               ) -> torch.Tensor:
    """x [..., K] → [..., N] bf16。w_fp8 为 [N, K] 行主序。"""
    lead, k = x.shape[:-1], x.shape[-1]
    flat = x.reshape(-1, k).contiguous()
    q, scale = quantize_tensorwise(flat)
    y = torch._scaled_mm(q, w_fp8.T, scale_a=scale, scale_b=w_scale.reshape(1),
                         out_dtype=torch.bfloat16)
    if isinstance(y, tuple):
        y = y[0]
    return y.view(*lead, w_fp8.shape[0])
