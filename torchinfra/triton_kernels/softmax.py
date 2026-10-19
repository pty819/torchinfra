import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (2, 4, 8, 16)],
    key=["N"],
)
@triton.jit
def _softmax_kernel(x_ptr, out_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride + offs, mask=mask,
                other=float("-inf")).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(out_ptr + row * stride + offs, y.to(out_ptr.dtype.element_ty),
             mask=mask)


def _next_pow2(n: int) -> int:
    return triton.next_power_of_2(n)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    x2 = x.view(-1, x.shape[-1])
    out = torch.empty_like(x2)
    rows, n = x2.shape
    grid = lambda meta: (rows,)
    _softmax_kernel[grid](x2, out, out.stride(0), n, BLOCK=_next_pow2(n))
    return out.view(x.shape)
