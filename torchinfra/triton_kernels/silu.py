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


@triton.jit
def _silu_mul_amax_kernel(GU, OUT, PART, cg, cu,
                          I: tl.constexpr, BLOCK: tl.constexpr):
    """GU [B, 2I] → OUT [B, I] + 每行 amax partial（fp8 融合量化用）。

    amax 取 bf16 舍入后的值，与单独跑 _row_amax bit-exact 一致。
    cg/cu 是 gate/up 段的 scale 修正（融合 GEMM 的 epilogue 只能带一个
    tensorwise scale=gate 的，up 段差 (su/sg) 倍，在 load 后 fp32 补乘）。"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < I
    base = row * 2 * I
    g = tl.load(GU + base + offs, mask=mask, other=0.0).to(tl.float32) * cg
    u = tl.load(GU + base + I + offs, mask=mask, other=0.0).to(tl.float32) * cu
    v = (g * tl.sigmoid(g) * u).to(OUT.dtype.element_ty)
    tl.store(OUT + row * I + offs, v, mask=mask)
    tl.store(PART + row, tl.max(tl.abs(v.to(tl.float32))))


def silu_mul_amax(gu: torch.Tensor, cg: float = 1.0, cu: float = 1.0):
    """gu [B, 2I] 连续（前半 gate、后半 up）→ ([B, I], partial [B] fp32)。"""
    B, two_i = gu.shape
    I = two_i // 2
    out = torch.empty(B, I, dtype=gu.dtype, device=gu.device)
    part = torch.empty(B, device=gu.device, dtype=torch.float32)
    _silu_mul_amax_kernel[(B,)](gu, out, part, cg, cu, I=I,
                                BLOCK=max(triton.next_power_of_2(I), 16))
    return out, part


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """gu [B, 2I] 连续 → [B, I]。"""
    B, two_i = gu.shape
    I = two_i // 2
    out = torch.empty(B, I, dtype=gu.dtype, device=gu.device)
    _silu_mul_kernel[(B,)](gu, out, I=I,
                           BLOCK=max(triton.next_power_of_2(I), 16))
    return out
