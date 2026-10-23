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


@triton.jit
def _rmsnorm_amax_kernel(x_ptr, w_ptr, out_ptr, part_ptr, stride, N, eps,
                         BLOCK: tl.constexpr):
    """rmsnorm + 顺带写每行 amax partial（fp8 融合量化用）。

    amax 取 bf16 舍入后的值（即后续量化 kernel 读回的同一 bit pattern），
    与单独跑 _row_amax 的结果 bit-exact 一致。配置锁死（无 autotune）：
    BLOCK 影响归约树，autotune 按耗时选配置会让输出随计时噪声漂移
    （最后一次 ulp），bit-exact 路径不能有这个自由度。N=1024 时单次迭代。"""
    row = tl.program_id(0)
    base = row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    rms = tl.rsqrt(tl.sum(acc, 0) / N + eps)
    amax = 0.0
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x * rms * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + base + offs, y, mask=mask)
        amax = tl.maximum(amax, tl.max(tl.abs(y.to(tl.float32))))
    tl.store(part_ptr + row, amax)


def triton_rmsnorm_amax(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """返回 (normed bf16, 每行 amax partial [M] fp32)。一趟读写。"""
    x = x.contiguous()
    shape = x.shape
    x2 = x.view(-1, shape[-1])
    out = torch.empty_like(x2)
    part = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
    _rmsnorm_amax_kernel[(x2.shape[0],)](x2, w, out, part, out.stride(0),
                                         shape[-1], eps,
                                         BLOCK=min(max(triton.next_power_of_2(
                                             shape[-1]), 16), 1024),
                                         num_warps=4)
    return out.view(shape), part


def triton_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
                   ) -> torch.Tensor:
    """函数式 RMSNorm：x [*, N] 连续化后整行一趟（与模块同 kernel）。"""
    x = x.contiguous()
    shape = x.shape
    x2 = x.view(-1, shape[-1])
    out = torch.empty_like(x2)
    _rmsnorm_kernel[(x2.shape[0],)](x2, w, out, out.stride(0),
                                    shape[-1], eps)
    return out.view(shape)


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


@triton.jit
def _fused_kernel(x_ptr, dx_ptr, w_ptr, res_ptr, norm_ptr, stride, N, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dx = tl.load(dx_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = (x + dx).to(res_ptr.dtype.element_ty)   # 对齐 torch：residual 先落 dtype
    r32 = r.to(tl.float32)                      # norm 读回的就是舍入后的值
    rms = tl.rsqrt(tl.sum(r32 * r32, 0) / N + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(res_ptr + base + offs, r, mask=mask)
    tl.store(norm_ptr + base + offs, (r32 * rms * w).to(norm_ptr.dtype.element_ty),
             mask=mask)


def fused_rmsnorm_residual(x: torch.Tensor, dx: torch.Tensor,
                           w: torch.Tensor, eps: float = 1e-6):
    """residual=(x+dx) 落 dtype；normed=rmsnorm(residual)*w。一趟读写。"""
    x, dx = x.contiguous(), dx.contiguous()
    shape = x.shape
    n = shape[-1]
    res = torch.empty_like(x)
    normed = torch.empty_like(x)
    x2, dx2 = x.view(-1, n), dx.view(-1, n)
    r2, n2 = res.view(-1, n), normed.view(-1, n)
    rows = x2.shape[0]
    block = max(triton.next_power_of_2(n), 16)
    _fused_kernel[(rows,)](x2, dx2, w, r2, n2, x2.stride(0), n, eps,
                           BLOCK=block)
    return res, normed
