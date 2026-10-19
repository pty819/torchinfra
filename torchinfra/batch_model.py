"""批量 decode 路径：per-request starts 的 rope + paged attention 组装。"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_decode_batch_kernel(x_ptr, cos_ptr, sin_ptr, starts_ptr,
                              N_HEADS: tl.constexpr, D: tl.constexpr,
                              HALF: tl.constexpr, stride_row):
    row = tl.program_id(0)          # 行主序 [B, heads, 1, D]
    b = row // N_HEADS
    pos = tl.load(starts_ptr + b)
    offs = tl.arange(0, HALF)
    base = row * stride_row
    x1 = tl.load(x_ptr + base + offs).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF + offs).to(tl.float32)
    cos = tl.load(cos_ptr + pos * D + offs)
    sin = tl.load(sin_ptr + pos * D + offs)
    tl.store(x_ptr + base + offs,
             (x1 * cos - x2 * sin).to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + base + HALF + offs,
             (x2 * cos + x1 * sin).to(x_ptr.dtype.element_ty))


def _compact4d(x: torch.Tensor) -> torch.Tensor:
    """[B, H, 1, D] 的假连续陷阱同 P2 rope：强制 stride(2)==D。"""
    x = x.contiguous()
    if x.stride(2) != x.shape[-1]:
        x = x.clone(memory_format=torch.contiguous_format)
    return x


def rope_decode_batch(q: torch.Tensor, k: torch.Tensor,
                      starts: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """原地旋转 q/k [B, heads, 1, D]；starts 为各请求位置（int32）。"""
    q, k = _compact4d(q), _compact4d(k)
    d = q.shape[-1]
    for x in (q, k):
        heads = x.shape[1]
        rows = x.numel() // d
        _rope_decode_batch_kernel[(rows,)](
            x, cos, sin, starts, heads, d, d // 2, x.stride(2))
    return q, k
