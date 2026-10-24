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


def quantize_from_partial(x: torch.Tensor, partial: torch.Tensor):
    """配合生产者 kernel 吐出的每行 amax partial：跳过全量 _row_amax 读。

    x [M, K] 连续，partial [M] 为 x 的逐行 amax（bf16 舍入后值）。
    返回与 quantize_tensorwise bit-exact 相同的 (q, scale)。"""
    m, k = x.shape
    scale = (partial.amax() / FP8_MAX).clamp(min=1e-12)
    q = torch.empty(m, k, device=x.device, dtype=torch.float8_e4m3fn)
    block = 1024 if k <= 1024 else 2048 if k <= 2048 else 4096
    _quant_e4m3[(m,)](x, q, scale, x.stride(0), k, BLOCK=block, num_warps=4)
    return q, scale.reshape(1)


def fp8_linear_q(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor,
                 partial: torch.Tensor) -> torch.Tensor:
    """fp8_linear 的融合版：激活的 amax 由生产者 kernel 顺带算好。

    partial 必须是 x 展平成 [-1, K] 后每行的 amax。"""
    lead, k = x.shape[:-1], x.shape[-1]
    flat = x.reshape(-1, k).contiguous()
    q, scale = quantize_from_partial(flat, partial)
    y = torch._scaled_mm(q, w_fp8.T, scale_a=scale, scale_b=w_scale.reshape(1),
                         out_dtype=torch.bfloat16)
    if isinstance(y, tuple):
        y = y[0]
    return y.view(*lead, w_fp8.shape[0])


def fp8_linear_q_multi(x: torch.Tensor,
                       ws: list[tuple[torch.Tensor, torch.Tensor]],
                       partial: torch.Tensor) -> list[torch.Tensor]:
    """同一激活 × 多个权重：只量化一次，喂多个 GEMM。

    per-projection scale（split_scales checkpoint）的配套——q/k/v 或 gate/up
    各自的权重 scale 不同，但激活侧仍是一个 tensorwise scale，没必要量化
    三遍。ws = [(w_fp8 [N,K], w_scale), ...]，各权重的 K 必须一致。"""
    lead, k = x.shape[:-1], x.shape[-1]
    flat = x.reshape(-1, k).contiguous()
    q, scale = quantize_from_partial(flat, partial)
    outs = []
    for w_fp8, w_scale in ws:
        y = torch._scaled_mm(q, w_fp8.T, scale_a=scale,
                             scale_b=w_scale.reshape(1),
                             out_dtype=torch.bfloat16)
        if isinstance(y, tuple):
            y = y[0]
        outs.append(y.view(*lead, w_fp8.shape[0]))
    return outs


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
