"""SiLU(gate) * up 二合一：gate/up 拼接权重（fuse.py）的配套激活 kernel。"""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(GU, OUT, I: tl.constexpr, BLOCK: tl.constexpr):
    """GU [B, 2I] 连续 → OUT [B, I] = silu(GU[:, :I]) * GU[:, I:]。"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < I
    base = row * 2 * I
    g = tl.load(GU + base + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + base + I + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + row * I + offs,
             (g * tl.sigmoid(g) * u).to(OUT.dtype.element_ty), mask=mask)


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """gu [B, 2I] 连续 → [B, I]。"""
    B, two_i = gu.shape
    I = two_i // 2
    out = torch.empty(B, I, dtype=gu.dtype, device=gu.device)
    _silu_mul_kernel[(B,)](gu, out, I=I,
                           BLOCK=max(triton.next_power_of_2(I), 16))
    return out
