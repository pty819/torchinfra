# Raw fp8 WMMA (16x16x128) throughput micro-bench via FlyDSL.
# Operands resident in registers, no memory traffic in the loop — measures
# the instruction rate ceiling on gfx1201, to compare against the 241 TF
# figure we previously measured with a Triton raw-dot bench (P14).

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from flydsl._mlir import ir

from flydsl_kernels.flash_attn_func_gfx1201 import (
    _pointer_to_llvm_ptr,
    _pointer_load,
    _pointer_store,
)
from flydsl_kernels.tensor_shim import _run_compiled

BLOCK = 256  # 8 waves x 32
ITERS = 4096
NACC = 16  # independent accumulator chains per wave


@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def fp8_wmma_kernel(OUT: fx.Pointer, n_iter: fx.Int32):
    o_ptr = _pointer_to_llvm_ptr(OUT)
    lane = fx.Index(gpu.thread_idx.x)

    # A/B fragments: v2i32 per lane (16x16 fp8 = 256B / 32 lanes = 8B/lane)
    # fill with a benign pattern; values don't matter for throughput
    a_frag = Vec.filled(2, 0x01010101, fx.Int32).ir_value()
    b_frag = Vec.filled(2, 0x02020202, fx.Int32).ir_value()

    v8f32 = Vec.make_type(8, fx.Float32)
    accs = [Vec.filled(8, 0.0, fx.Float32).ir_value() for _ in range(NACC)]

    for i, acc in zip(range_constexpr_count(NACC), accs):
        pass  # (accumulators held in list; python-level structure only)

    # dependent loop: each chain accumulates serially, chains are independent
    for _ in range(n_iter):
        for k in range_constexpr(NACC):
            accs[k] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                v8f32, a_frag, b_frag, accs[k]
            ).result

    # prevent DCE: store first element of each accumulator
    out_base = lane * NACC
    for k in range_constexpr(NACC):
        val = Vec(accs[k])[0]
        gep_out = o_ptr  # simplified below
        from flydsl.expr import buffer_ops

        idx = arith.index_cast(T.i64, out_base + fx.Index(k))
        ptr = buffer_ops.get_element_ptr(o_ptr, idx, elem_type=T.f32)
        _pointer_store(_raw(fx.Float32(val)), ptr)


def range_constexpr_count(n):
    from flydsl.expr import range_constexpr

    return range_constexpr(n)


@flyc.jit
def launch_fp8_wmma(OUT: fx.Pointer, n_iter: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    launcher = fp8_wmma_kernel(OUT, n_iter)
    launcher.launch(grid=(224, 1, 1), block=(BLOCK, 1, 1), stream=stream)


def main():
    out = torch.zeros(224 * BLOCK * NACC, dtype=torch.float32, device="cuda")
    launch_fp8_wmma.compile_hints = {}

    def _ptr(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    import time

    t0 = time.time()
    _run_compiled(launch_fp8_wmma, _ptr(out), ITERS)
    torch.cuda.synchronize()
    print(f"compile+run: {time.time()-t0:.1f}s")

    # timing with events
    for _ in range(3):
        _run_compiled(launch_fp8_wmma, _ptr(out), ITERS)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    _run_compiled(launch_fp8_wmma, _ptr(out), ITERS)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)

    flops = 224 * (BLOCK // 32) * ITERS * NACC * 16 * 16 * 16 * 2
    print(f"time {ms:.2f}ms -> {flops/ms/1e9:.1f} TFLOP/s (fp8 wmma raw)")
    print(f"out[0..3] = {out[:4].tolist()}")


if __name__ == "__main__":
    main()
