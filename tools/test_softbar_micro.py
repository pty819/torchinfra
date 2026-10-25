import sys, os
sys.path.insert(0, "~/torchinfra")
from flydsl.expr.enum import AtomicOrdering as _AO
ORD_ARR = getattr(_AO, os.environ.get("ORD_ARR", "SeqCst"))
ORD_POLL = getattr(_AO, os.environ.get("ORD_POLL", "SeqCst"))
ORD_DATA = getattr(_AO, os.environ.get("ORD_DATA", "SeqCst"))
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import llvm as fllv
from flydsl.expr.enum import AtomicOrdering
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from flydsl.expr.typing import T as TT
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl_kernels._buffer_ops_impl import create_llvm_ptr, extract_base_index
from flydsl_kernels.flydsl_compat import _install_buffer_ops
from flydsl.expr import buffer_ops

GROUPS, WAVES, LANES = 2, 8, 32
WAVES_PER_GROUP = WAVES // GROUPS
ROUNDS = 64
ARRIVES = WAVES_PER_GROUP * LANES


def _lptr(ptr, result_type):
    return _llvm.LoadOp(result_type, ptr).result


def build():
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="softbar_smem")
    off = allocator._align(allocator.ptr, 8)
    allocator.ptr = off + 8 + 256 * 4  # counter[2] + data[256] (i32)

    @flyc.kernel(known_block_size=[WAVES * LANES, 1, 1])
    def k(OUT: fx.Pointer, rounds: fx.Int32):
        base = allocator.get_base()
        cnt_mr = SmemPtr(base, off, TT.i32, shape=(GROUPS,)).get()
        data_mr = SmemPtr(base, off + 8, TT.i32, shape=(256,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(LANES)
        wave = tid // fx.Index(LANES)
        grp_i = wave // fx.Index(WAVES_PER_GROUP)          # 0/1
        grp = fx.Int32(grp_i)

        # 计数器指针：base + grp*4B，地址空间 3
        cp = create_llvm_ptr(extract_base_index(cnt_mr, address_space=3) + grp_i * fx.Index(4), 3)

        # 清零 + 一次硬件 barrier 保证可见
        _memref.store(_raw(fx.Int32(0)), cnt_mr, [_raw(grp_i)])
        _memref.store(_raw(fx.Int32(0)), data_mr, [_raw(grp_i)])
        gpu.barrier()

        # data 扩为 256 槽（每 lane 一槽）；每轮：写自己的槽 -> 软 barrier ->
        # 读同组“下一个 wave”同 lane 的槽（真正的跨 wave 可见性检验，且
        # 循环旋转无关：值 = 1000*round + slot，与轮次单调绑定）
        err = fx.Int32(0)
        my_slot = tid  # 0..255（两组各占一半天然分开：grp0=0..127, grp1=128..255）
        peer_wave = (wave + fx.Index(1)) % fx.Index(WAVES_PER_GROUP)
        peer_slot = (grp_i * fx.Index(WAVES_PER_GROUP) + peer_wave) * fx.Index(LANES) + lane
        for i in range(0, fx.Index(rounds), 1):
            rnd = fx.Int32(i) + fx.Int32(1)
            _memref.store(_raw(rnd), data_mr, [_raw(my_slot)])
            fllv.memory_fence(syncscope="workgroup", ordering=_AO.SeqCst)
            if lane == fx.Index(0):
                fllv.atomic_add(cp, fx.Int32(1),
                                ordering=ORD_ARR, syncscope="workgroup")
            tgt = fx.Int32(WAVES_PER_GROUP) * rnd
            cur = fx.Int32(_memref.load(cnt_mr, [_raw(grp_i)]))
            while arith.cmpi(arith.CmpIPredicate.slt, _raw(fx.Int32(cur)), _raw(tgt)):
                cur = fx.Int32(_memref.load(cnt_mr, [_raw(grp_i)]))
            pep = create_llvm_ptr(extract_base_index(data_mr, address_space=3) + peer_slot * fx.Index(4), 3)
            got = fx.Int32(fllv.atomic_add(pep, fx.Int32(0), ordering=AtomicOrdering.Acquire, syncscope="workgroup"))
            bad = fx.Int32(ArithValue(arith.cmpi(arith.CmpIPredicate.ne, _raw(got), _raw(rnd))
                                      ).select(fx.Int32(1), fx.Int32(0)))
            err = err + bad

        # OUT[tid] = err（全局 i32 写）
        o_ptr = fx.Pointer  # noqa
        from flydsl_kernels.flash_attn_fp8_gfx1201 import _pointer_to_llvm_ptr
        op = _pointer_to_llvm_ptr(OUT)
        gep = buffer_ops.get_element_ptr(op, fx.Int64(tid), elem_type=T.i32)
        _llvm.StoreOp(_raw(err), gep)

    @flyc.jit
    def launch(OUT: fx.Pointer, rounds: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        from flydsl.compiler.kernel_function import CompilationContext
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        k(OUT, rounds).launch(grid=(1, 1, 1), block=(WAVES * LANES, 1, 1), stream=stream)

    launch.compile_hints = {"fast_fp_math": True, "unsafe_fp_math": True,
                            "llvm_options": {"enable-post-misched": False}}
    return launch


if __name__ == "__main__":
    import torch, time
    _REP = int(os.environ.get("REP", "0"))
    launch = build()
    out = torch.zeros((256,), dtype=torch.int32, device="cuda")
    t0 = time.perf_counter()
    import sys as _s
    ROUNDS = int(_s.argv[1]) if len(_s.argv) > 1 else ROUNDS
    cf = flyc.compile(launch, flyc.from_c_void_p(fx.Int32, out.data_ptr()), ROUNDS)
    torch.cuda.synchronize()
    print(f"运行 {time.perf_counter()-t0:.2f}s（含编译）")
    if _REP:
        ptr_x = flyc.from_c_void_p(fx.Int32, out.data_ptr())
        fails = 0
        import time as _t
        for _ in range(_REP):
            out.zero_()
            cf(ptr_x, ROUNDS, None)
            torch.cuda.synchronize()
            fails += int((out != 0).sum() > 0)
        print(f"REP={_REP} 失败 {fails} 次")
        import sys as _s2; _s2.exit(0)
    o = out.cpu().numpy()
    bad = [(i, int(v)) for i, v in enumerate(o) if v]
    print(f"非0数={len(bad)} wave分布={sorted(set(i//32 for i,_ in bad))} err值={sorted(set(v for _,v in bad))}")
    print("失败样例:", bad[:10])
    out2 = torch.full((256,), -1, dtype=torch.int32, device="cuda")
    ptr2 = flyc.from_c_void_p(fx.Int32, out2.data_ptr())
    t0 = time.perf_counter()
    for _ in range(20):
        cf(ptr2, ROUNDS, None)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    print(f"稳态 {dt*1e6:.1f}us/kernel -> {dt/ROUNDS*1e9:.0f}ns/软barrier轮（含读写校验）")
