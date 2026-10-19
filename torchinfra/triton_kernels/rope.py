import torch
import triton
import triton.language as tl
from torchinfra.modules import RotaryEmbedding


@triton.jit(do_not_specialize=["S", "start", "stride_row"])
def _rope_inplace_kernel(x_ptr, cos_ptr, sin_ptr, S, start, D: tl.constexpr,
                         HALF: tl.constexpr, stride_row):
    row = tl.program_id(0)
    pos = start + row % S   # 行主序 [b, h, s, d] 中 s 是最内层非 d 维
    offs = tl.arange(0, HALF)
    base = row * stride_row
    x1 = tl.load(x_ptr + base + offs).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF + offs).to(tl.float32)
    cos = tl.load(cos_ptr + pos * D + offs)   # 表 [max_pos, D] 双份半区，取前半
    sin = tl.load(sin_ptr + pos * D + offs)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    tl.store(x_ptr + base + offs, o1.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + base + HALF + offs, o2.to(x_ptr.dtype.element_ty))


class TritonRoPE(RotaryEmbedding):
    @staticmethod
    def _compact(x: torch.Tensor) -> torch.Tensor:
        # 陷阱：size-1 维的 stride 不参与 torch 的连续性判定，s==1 的
        # transposed 视图（stride(2)=H*D）会被 contiguous()/clone() 原样
        # 保留，而 kernel 按 stride(2)==d 的行主序寻址 → 越界。
        x = x.contiguous()
        if x.stride(2) != x.shape[-1]:
            x = x.clone(memory_format=torch.contiguous_format)
        return x

    def apply_qk(self, q: torch.Tensor, k: torch.Tensor, start: int):
        q = self._compact(q)
        k = self._compact(k)
        d = q.shape[-1]
        for x, s_len in ((q, q.shape[2]), (k, k.shape[2])):
            rows = x.numel() // d
            _rope_inplace_kernel[(rows,)](
                x, self.cos, self.sin, s_len, start, d, d // 2, x.stride(2))
        return q, k
