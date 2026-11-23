# micro v3: 真·软 barrier 配方实验（自旋带 keepalive 保活，循环内有活的全局加载）
# 结构镜像 wsplit：for+yield 携带(10 值) + 双计数器(ci_data/ci_done) + leader 到达 + 2 barrier/轮
# 排序参数化：ARR∈{rel,acqrel,mono} SPIN∈{plain,atomacq} FENCE∈{none,seqcst}
import sys, os
ARR = os.environ.get("ARR", "rel")        # arrive 原子排序
SPIN = os.environ.get("SPIN", "plain")    # 自旋读法
FENCE = os.environ.get("FENCE", "none")   # 额外 fence
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr import llvm as fllv
from flydsl.expr.enum import AtomicOrdering as AO
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl_kernels._buffer_ops_impl import create_llvm_ptr, extract_base_index
from flydsl_kernels.flash_attn_fp8_gfx1201 import _pointer_to_llvm_ptr
from flydsl.expr import buffer_ops

GROUPS, WAVES, LANES = 2, 8, 32
WPG = WAVES // GROUPS
ROUNDS = 64
DATA_OFF = 16
KA_OFF = DATA_OFF + 2 * 4 * 32 * 4
LDS_TOTAL = KA_OFF + 4

ORD_MAP = {"rel": AO.Release, "acqrel": AO.AcqRel, "mono": None}


def build():
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="m3_smem")
    off = allocator._align(allocator.ptr, 16)
    allocator.ptr = off + LDS_TOTAL

    @flyc.kernel(known_block_size=[WAVES * LANES, 1, 1])
    def k(OUT: fx.Pointer, OUT2: fx.Pointer, G: fx.Pointer, rounds: fx.Int32):
        base = allocator.get_base()
        cnt_mr = SmemPtr(base, off, T.i32, shape=(4,)).get()
        data_mr = SmemPtr(base, off + DATA_OFF, T.i32, shape=(2 * 4 * 32,)).get()
        ka_mr = SmemPtr(base, off + KA_OFF, T.i32, shape=(1,)).get()

        g_ptr = _pointer_to_llvm_ptr(G)

        bid = fx.Index(gpu.block_idx.x)
        gtid = bid * fx.Index(WAVES * LANES) + fx.Index(gpu.thread_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(LANES)
        wave = tid // fx.Index(LANES)
        grp_i = wave // fx.Index(WPG)
        is_g0 = arith.cmpi(arith.CmpIPredicate.eq, _raw(fx.Int32(grp_i)), _raw(fx.Int32(0)))
        wig = wave % fx.Index(WPG)
        peer_wig = (wig + fx.Index(1)) % fx.Index(WPG)
        my_slot = grp_i * fx.Index(WPG * LANES) + wig * fx.Index(LANES) + lane
        peer_slot = grp_i * fx.Index(WPG * LANES) + peer_wig * fx.Index(LANES) + lane

        def cnt_ptr(ci):
            return create_llvm_ptr(
                extract_base_index(cnt_mr, address_space=3) + ci * fx.Index(4), 3)

        def soft_wait(ci, tgt):
            cur = fx.Int32(_memref.load(cnt_mr, [_raw(ci)]))
            while arith.cmpi(arith.CmpIPredicate.slt, _raw(cur), _raw(tgt)):
                cur = fx.Int32(_memref.load(cnt_mr, [_raw(ci)]))
            if FENCE == "seqcst":
                fllv.memory_fence(syncscope="workgroup", ordering=AO.SeqCst)
            _memref.store(_raw(cur), ka_mr, [_raw(fx.Index(0))])  # keepalive

        def soft_wait_a(ci, tgt):
            _ord = AO.Acquire if SPIN == "atomacq" else AO.Monotonic
            cur = fx.Int32(fllv.atomic_add(cnt_ptr(ci), fx.Int32(0),
                                           ordering=_ord, syncscope="workgroup"))
            while arith.cmpi(arith.CmpIPredicate.slt, _raw(cur), _raw(tgt)):
                cur = fx.Int32(fllv.atomic_add(cnt_ptr(ci), fx.Int32(0),
                                               ordering=_ord, syncscope="workgroup"))
            _memref.store(_raw(cur), ka_mr, [_raw(fx.Index(0))])  # keepalive

        def soft_arrive(ci):
            if lane == fx.Index(0):
                if ARR == "mono":
                    fllv.atomic_add(cnt_ptr(ci), fx.Int32(1))
                else:
                    fllv.atomic_add(cnt_ptr(ci), fx.Int32(1),
                                    ordering=ORD_MAP[ARR], syncscope="workgroup")

        _wait_fn = soft_wait if SPIN == "plain" else soft_wait_a
        NOBAR = os.environ.get("NOBAR", "0") == "1"

        def soft_bar(ci, tgt):
            if not NOBAR:
                soft_arrive(ci)
                _wait_fn(ci, tgt)

        _memref.store(_raw(fx.Int32(0)), cnt_mr, [_raw(fx.Index(0))])
        _memref.store(_raw(fx.Int32(0)), cnt_mr, [_raw(fx.Index(1))])
        _memref.store(_raw(fx.Int32(0)), cnt_mr, [_raw(fx.Index(2))])
        _memref.store(_raw(fx.Int32(0)), cnt_mr, [_raw(fx.Index(3))])
        gpu.barrier()

        # 循环内全局加载：G[gtid], G[gtid+1]（混入 acc 保活）
        def gld(i):
            gep = buffer_ops.get_element_ptr(g_ptr, fx.Int64(i), elem_type=T.i32)
            return fx.Int32(_llvm.LoadOp(T.i32, gep).result)

        _init = ([_raw(fx.Int32(0)), _raw(fx.Float32(0.0))]
                 + [_raw(fx.Float32(0.0)) for _ in range(8)]
                 + [_raw(fx.Int32(0))])   # acc, l, o0..o7, err
        loop_results = _init
        for it_i, iargs in range(0, fx.Index(rounds), 1, init=_init):
            acc = fx.Int32(iargs[0])
            l = fx.Float32(iargs[1])
            o0_ = iargs[2]; o1_ = iargs[3]; o2_ = iargs[4]; o3_ = iargs[5]
            o4_ = iargs[6]; o5_ = iargs[7]; o6_ = iargs[8]; o7_ = iargs[9]
            err = fx.Int32(iargs[10])
            it = fx.Int32(it_i)

            g0 = gld(gtid + it_i)                    # 地址随迭代变化，防 LICM 提升
            g1 = gld(gtid + it_i + fx.Index(1))
            acc = acc + (g0 & fx.Int32(1)) + (g1 & fx.Int32(1))

            if not NOBAR:
                soft_wait(grp_i, fx.Int32(WPG) * (it))
            val = fx.Int32(1000) * (it + fx.Int32(1)) + fx.Int32(lane)
            _memref.store(_raw(val), data_mr, [_raw(my_slot)])
            soft_bar(grp_i, fx.Int32(WPG) * (it + fx.Int32(1)))
            got = fx.Int32(_memref.load(data_mr, [_raw(peer_slot)]))
            bad = fx.Int32(ArithValue(arith.cmpi(
                arith.CmpIPredicate.ne, _raw(got), _raw(val))).select(
                fx.Int32(1), fx.Int32(0)))
            err = err + bad
            gf = fx.Float32(got)
            l = arith.addf(_raw(l), _raw(arith.mulf(_raw(gf), _raw(fx.Float32(0.001)))))
            os_here = [o0_, o1_, o2_, o3_, o4_, o5_, o6_, o7_]
            for j in range_constexpr(8):
                os_here[j] = arith.addf(
                    _raw(os_here[j]),
                    _raw(arith.mulf(_raw(gf), _raw(fx.Float32(0.01)))))
            o0_ = os_here[0]; o1_ = os_here[1]; o2_ = os_here[2]; o3_ = os_here[3]
            o4_ = os_here[4]; o5_ = os_here[5]; o6_ = os_here[6]; o7_ = os_here[7]
            loop_results = yield [_raw(acc), _raw(l),
                                  o0_, o1_, o2_, o3_, o4_, o5_, o6_, o7_, _raw(err)]

        acc_f = fx.Int32(loop_results[0])
        err_f = fx.Int32(loop_results[10])
        o_ptr = _pointer_to_llvm_ptr(OUT)
        gep = buffer_ops.get_element_ptr(o_ptr, fx.Int64(gtid), elem_type=T.i32)
        _llvm.StoreOp(_raw(err_f * fx.Int32(1000) + acc_f), gep)
        o2_ptr = _pointer_to_llvm_ptr(OUT2)
        gep2 = buffer_ops.get_element_ptr(o2_ptr, fx.Int64(gtid), elem_type=T.f32)
        _llvm.StoreOp(_raw(fx.Float32(loop_results[1])), gep2)

    @flyc.jit
    def launch(OUT: fx.Pointer, OUT2: fx.Pointer, G: fx.Pointer, rounds: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        from flydsl.compiler.kernel_function import CompilationContext
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        k(OUT, OUT2, G, rounds).launch(grid=(GRID, 1, 1), block=(WAVES * LANES, 1, 1),
                                       stream=stream)

    launch.compile_hints = {"fast_fp_math": True, "llvm_options": {"enable-post-misched": False}}
    return launch


if __name__ == "__main__":
    import numpy as np
    import torch
    GRID = int(os.environ.get("GRID", "1"))
    REP = int(os.environ.get("REP", "20"))
    R = int(sys.argv[1]) if len(sys.argv) > 1 else ROUNDS
    launch = build()
    N = GRID * 256
    out = torch.zeros((N,), dtype=torch.int32, device="cuda")
    out2 = torch.zeros((N,), dtype=torch.float32, device="cuda")
    g = (torch.arange(N + 300, dtype=torch.int32, device="cuda") * 3 + 7) % 251
    g0 = g.cpu().numpy()
    cf = flyc.compile(launch,
                      flyc.from_c_void_p(fx.Int32, out.data_ptr()),
                      flyc.from_c_void_p(fx.Float32, out2.data_ptr()),
                      flyc.from_c_void_p(fx.Int32, g.data_ptr()),
                      R)
    torch.cuda.synchronize()
    exp_l = {lane: sum((1000 * it + lane) * 0.001 for it in range(1, R + 1)) for lane in range(32)}
    fails = 0
    for rep in range(REP):
        out.fill_(-1)
        out2.fill_(float("nan"))
        cf(flyc.from_c_void_p(fx.Int32, out.data_ptr()),
           flyc.from_c_void_p(fx.Float32, out2.data_ptr()),
           flyc.from_c_void_p(fx.Int32, g.data_ptr()), R, None)
        torch.cuda.synchronize()
        e = out.cpu().numpy(); f2 = out2.cpu().numpy()
        bad = 0
        for b in range(GRID):
            for w in range(8):
                seg = slice(b * 256 + w * 32, b * 256 + w * 32 + 32)
                ev = e[seg].astype(np.int64)
                exp_acc = np.array([
                    sum(((g0[b * 256 + i + it] & 1) + (g0[b * 256 + i + it + 1] & 1))
                        for it in range(R)) for i in range(32)], dtype=np.int64)
                nerr = int((ev // 1000 != 0).sum())
                nacc = int(((ev % 1000) != exp_acc).sum())
                if nerr or nacc:
                    bad += 1
                    if rep < 2 and b == 0 and w < 2:
                        print(f"    w{w}: errLanes={nerr} accLanes={nacc} "
                              f"badlane_idx={np.nonzero((ev%1000)!=exp_acc)[0].tolist()} "
                              f"got={ev[ev%1000!=exp_acc][:4].tolist()}")
                expl = np.array([exp_l[i % 32] for i in range(32)])
                if (~np.isclose(f2[seg], expl, rtol=1e-4, atol=1e-2)).any():
                    bad += 1
        if bad:
            fails += 1
            if fails <= 3:
                print(f"  rep{rep}: bad waves={bad}")
    print(f"ARR={ARR} SPIN={SPIN} FENCE={FENCE} GRID={GRID}: {fails}/{REP} 失败")
