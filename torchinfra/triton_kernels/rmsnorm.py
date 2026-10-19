import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w)
             for b in (1024, 2048) for w in (4, 8)],
    key=["N"],
)
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride, N, eps,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    rms = tl.rsqrt(tl.sum(acc, 0) / N + eps)
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rms * w
        tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty),
                 mask=mask)


class TritonRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        shape = x.shape
        x2 = x.view(-1, shape[-1])
        out = torch.empty_like(x2)
        rows, n = x2.shape
        grid = lambda meta: (rows,)
        _rmsnorm_kernel[grid](x2, self.weight, out, out.stride(0), n,
                              self.eps)
        return out.view(shape)
