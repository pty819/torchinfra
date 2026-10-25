# 极小对照：for + yield 携带 acc，循环内变地址全局加载。acc 应 = rounds。
import sys, os
sys.path.insert(0, "~/torchinfra")
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.runtime.device import get_rocm_arch
from flydsl_kernels.flash_attn_fp8_gfx1201 import _pointer_to_llvm_ptr
from flydsl.expr import buffer_ops
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

GRID = 256


def build():
    get_rocm_arch()

    @flyc.kernel(known_block_size=[256, 1, 1])
    def k(OUT: fx.Pointer, G: fx.Pointer, rounds: fx.Int32):
        g_ptr = _pointer_to_llvm_ptr(G)
        gtid = fx.Index(gpu.block_idx.x) * fx.Index(256) + fx.Index(gpu.thread_idx.x)

        def gld(i):
            gep = buffer_ops.get_element_ptr(g_ptr, fx.Int64(i), elem_type=T.i32)
            return fx.Int32(_llvm.LoadOp(T.i32, gep).result)

        _init = [_raw(fx.Int32(0))]
        loop_results = _init
        for it_i, iargs in range(0, fx.Index(rounds), 1, init=_init):
            acc = fx.Int32(iargs[0])
            g0 = gld(gtid + it_i)
            g1 = gld(gtid + it_i + fx.Index(1))
            acc = acc + (g0 & fx.Int32(1)) + (g1 & fx.Int32(1))
            loop_results = yield [_raw(acc)]
        _lr = loop_results[0] if isinstance(loop_results, (list, tuple)) else loop_results
        acc_f = fx.Int32(_lr)
        o_ptr = _pointer_to_llvm_ptr(OUT)
        gep = buffer_ops.get_element_ptr(o_ptr, fx.Int64(gtid), elem_type=T.i32)
        _llvm.StoreOp(_raw(acc_f), gep)

    @flyc.jit
    def launch(OUT: fx.Pointer, G: fx.Pointer, rounds: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        k(OUT, G, rounds).launch(grid=(GRID, 1, 1), block=(256, 1, 1), stream=stream)

    launch.compile_hints = {"fast_fp_math": True, "llvm_options": {"enable-post-misched": False}}
    return launch


if __name__ == "__main__":
    import numpy as np
    import torch
    R = 64
    N = GRID * 256
    launch = build()
    out = torch.zeros((N,), dtype=torch.int32, device="cuda")
    g = (torch.arange(N + 300, dtype=torch.int32, device="cuda") * 3 + 7) % 251
    g0 = g.cpu().numpy()
    cf = flyc.compile(launch, flyc.from_c_void_p(fx.Int32, out.data_ptr()),
                      flyc.from_c_void_p(fx.Int32, g.data_ptr()), R)
    torch.cuda.synchronize()
    e = out.cpu().numpy()
    exp = np.array([sum(((g0[i + it] & 1) + (g0[i + it + 1] & 1)) for it in range(R))
                    for i in range(N)])
    bad = np.nonzero(e != exp)[0]
    print(f"bad={len(bad)}/{N}")
    if len(bad):
        print("bad gtid:", bad[:10].tolist())
        print("got:", e[bad[:6]].tolist(), "exp:", exp[bad[:6]].tolist())
        print("lane分布:", sorted(set(int(b % 32) for b in bad[:200])))
