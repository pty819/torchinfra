import torch
import triton
import triton.language as tl
from torchinfra.modules import RotaryEmbedding


@triton.jit
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
    def apply_qk(self, q: torch.Tensor, k: torch.Tensor, start: int):
        # transposed 视图内存序是 [b, s, h, d]，kernel 按行主序 [b, h, s, d] 处理，
        # 故先 contiguous（vLLM 用显式 stride 索引省掉这次拷贝，教学版从简）
        q = q.contiguous()
        k = k.contiguous()
        d = q.shape[-1]
        for x, s_len in ((q, q.shape[2]), (k, k.shape[2])):
            rows = x.numel() // d
            _rope_inplace_kernel[(rows,)](
                x, self.cos, self.sin, s_len, start, d, d // 2, x.stride(2))
        return q, k
