# ARCHIVED: warp-split v2 experiment; production uses flash_attn_fp8_gfx1201.py.
# wsplit2: 成对展开 + K/V 双缓冲 LDS + 任意 S 的 P 掩码
#
# 相对 wsplit(c656b12) 的三个结构变化：
#
# 1) 成对展开组循环（治 LDS base 悬崖 + 藏全局延迟）
#    每组 2 个 LDS 槽（slotA/slotB），槽基址在循环外算好（组内循环不变量，
#    只有跨组差一个运行期常量——按 flydsl-double-buffer-lds-base-cliff
#    技能，悬崖只咬循环内变化的基址）。稳态流水：
#      wait doneA(it-1) → store A(it) → bar dataA →
#      wait doneB(it-1) → store B(it) → bar dataB →
#      预取 pair it+1 的 4 组全局 load（在 compute A/B 整个期间在飞）→
#      compute A → arrive doneA → compute B → arrive doneB
#    B 的 LDS 暂存在 compute A 期间落定；pair it+1 的全局 load 延迟被
#    两倍 tile 计算盖住（旧版每 tile 只有等待间隙可盖）。
#
# 2) 计数器按槽分离（每组 4 个：dataA/doneA/dataB/doneB，共 8）
#    B 槽的等待目标与 A 槽独立，store B 不必等 compute A。
#
# 3) 任意 S 正确性：softmax 前按 seq 边界逐元素掩 s_raw
#    （映射：s_raw[half][r] ↔ k 行 = tile + half*16 + klane*8 + r，
#     由 lsum 的 peer_xor16 分母结构 + legacy 同公式交叉验证）。
#    掩成 -1e30 → p 精确为 0，lsum/GEMM2 零贡献。副作用：完全越界的
#    B tile（myn 奇数时的尾对）数学上成为空操作——循环体零分支、
#    全部 barrier 均匀到达（避开 while 嵌套被改写器丢弃的坑）。
#
# 4) NQ/NKV ∈ {4, 1}：比值 1（MHA，qwen21dit 32/32）时 q tile 64 行，
#    wave%4 选行块；比值 4（GQA，Nemotron 32/8）保持旧行为 16 行选头。
#    K/V 暂存摊销两档相同（都是 64 行计算量/tile）。
#
# LDS 布局（总 35364B < 64KB）：
#   cnt[8]@0 | g0: KA@32 VA@4256 KB@8864 VB@12888 | g1: 同构 @17696
#   | ka@35360 ；merge scratch 复用 g0 区起点 @32（33280B ≤ 35328B，
#   循环结束后 K/V 槽已死）
#
# 配方不变：软屏障 Monotonic 自旋 + Release 到达 + LDS keepalive，
# 零 fence；fast_fp_math + llvm enable-post-misched:False。

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import llvm as fllv
from flydsl.expr import math as fmath
from flydsl.expr.enum import AtomicOrdering
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, memref as _memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.expr import buffer_ops

from .tensor_shim import _run_compiled
from ._buffer_ops_impl import create_llvm_ptr, extract_base_index
from .flash_attn_fp8_gfx1201 import _pointer_to_llvm_ptr

WAVES = 8
WPG = 4
WARP = 32
K_STRIDE = 132
V_STRIDE = 36
K_TILE = 32 * K_STRIDE
V_TILE = 128 * V_STRIDE
SLOT = K_TILE + V_TILE                 # 8832B：一个 K+V 槽
LDS_G0 = 32                            # cnt[8] 之后，128B 对齐
GRP_STRIDE = 2 * SLOT                  # 每组两槽
KA_OFF = LDS_G0 + 2 * GRP_STRIDE       # 35360
LDS_TOTAL = KA_OFF + 4
MERGE_F32_PER_WAVE = 32 + 2048
MERGE_BYTES = MERGE_F32_PER_WAVE * 4 * 4   # 33280 ≤ GRP 区 35328，循环后复用


def build_flash_attn_fp8_wsplit2_module(
    num_heads,
    head_dim,
    num_kv_heads=None,
    waves_per_eu=None,
    flat_work_group_size=WAVES * WARP,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    assert head_dim == 128
    nkv = num_kv_heads if num_kv_heads is not None else num_heads
    ratio = num_heads // nkv
    assert ratio in (1, 4), "wsplit2 只支持 NQ/NKV = 4 (GQA) 或 1 (MHA)"
    NQ_PER_KV = ratio
    # 每块 q 行数：GQA4 时 4 个 wave = 4 个头 × 16 行 → tile 16 行；
    # MHA1 时 4 个 wave = 同头 4 个行块 → tile 64 行
    Q_TILE_ROWS = 16 if NQ_PER_KV == 4 else 64
    HEAD_DIM = head_dim
    STRIDE_TOKEN = num_heads * HEAD_DIM
    BLOCK_SIZE = flat_work_group_size
    VEC = 16
    TPR = HEAD_DIM // VEC

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="wsplit2_smem")
    _off = allocator._align(allocator.ptr, 16)
    allocator.ptr = _off + LDS_TOTAL   # 不推进 = LDS 分配 0 字节（全越界）

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_fp8_wsplit2_kernel(
        Q: fx.Pointer, K: fx.Pointer, V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32, seq_pad: fx.Int32, SCL: fx.Pointer,
    ):
        et = T.i8
        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        o_ptr = _pointer_to_llvm_ptr(O)
        scl_ptr = _pointer_to_llvm_ptr(SCL)

        def ld_f32(i):
            gep = buffer_ops.get_element_ptr(scl_ptr, fx.Int64(i), elem_type=T.f32)
            return fx.Float32(_llvm.LoadOp(ir.F32Type.get(), gep).result)

        qk_scale_log2e = ld_f32(0)
        scale_v = ld_f32(1)
        fm = arith.FastMathFlags.fast

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result

        v8f32_t = Vec.make_type(8, fx.Float32)
        v8i8_t = Vec.make_type(8, fx.Int8)
        v16i8_t = Vec.make_type(16, fx.Int8)

        def to_v2i32(x):
            return Vec(x).bitcast(fx.Int32)

        def wmma_fp8(a, b, c):
            return rocdl.wmma_f32_16x16x16_fp8_fp8(v8f32_t, _raw(a), _raw(b), c).result

        def pack_fp8(fs8):
            def pk(lo, hi, old, sel):
                return rocdl.cvt_pk_fp8_f32(T.i32, _raw(fx.Float32(lo)),
                                            _raw(fx.Float32(hi)), _raw(old), sel)
            d0 = pk(fs8[0], fs8[1], fx.Int32(0), 0)
            d0 = pk(fs8[2], fs8[3], d0, 1)
            d1 = pk(fs8[4], fs8[5], fx.Int32(0), 0)
            d1 = pk(fs8[6], fs8[7], d1, 1)
            return Vec.from_elements([d0, d1], fx.Int32).ir_value()

        seq_len_v = fx.Index(seq_len)
        s_pad_v = fx.Index(seq_pad)

        lds = SmemPtr(allocator.get_base(), 0, T.i8, shape=(LDS_TOTAL,)).get()
        cnt_mr = SmemPtr(allocator.get_base(), 0, T.i32, shape=(8,)).get()
        ka_mr = SmemPtr(allocator.get_base(), KA_OFF, T.i32, shape=(1,)).get()
        mrg = SmemPtr(allocator.get_base(), LDS_G0, T.f32,
                      shape=(MERGE_F32_PER_WAVE * 4,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(WARP)
        lane16 = lane % fx.Index(16)
        klane = lane // fx.Index(16)
        wave = tid // fx.Index(WARP)
        grp_i = wave // fx.Index(WPG)
        grp = fx.Int32(grp_i)
        is_g0 = arith.cmpi(arith.CmpIPredicate.eq, _raw(grp), _raw(fx.Int32(0)))

        block_id = fx.Index(gpu.block_idx.x)
        kv_head_idx = block_id % fx.Index(nkv)
        bqt = block_id // fx.Index(nkv)
        num_q_tiles = (seq_len_v + fx.Index(Q_TILE_ROWS - 1)) // fx.Index(Q_TILE_ROWS)
        q_tile_idx = bqt % num_q_tiles
        batch_idx = bqt // num_q_tiles
        q_start = q_tile_idx * fx.Index(Q_TILE_ROWS)

        if const_expr(NQ_PER_KV == 4):
            q_head = fx.Index(kv_head_idx * NQ_PER_KV) + (wave % fx.Index(WPG))
            q_row = q_start + lane16
        else:
            q_head = fx.Index(kv_head_idx)
            q_row = q_start + (wave % fx.Index(WPG)) * fx.Index(16) + lane16
        q_in_b = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        q_row_safe = fx.Index(ArithValue(q_in_b).select(q_row, fx.Index(0)))
        c_z8i8 = Vec.filled(8, 0, fx.Int8).ir_value()
        c_z8f = Vec.filled(8, 0.0, fx.Float32)
        c_ninf = fx.Float32(float("-inf"))
        c_0f = fx.Float32(0.0)
        c_1f = fx.Float32(1.0)
        c_448 = fx.Float32(448.0)
        c_msk = fx.Float32(-1.0e30)

        def qgi(h, s, c):
            return ((batch_idx * fx.Index(num_heads) + h) * s_pad_v + s) * HEAD_DIM + c

        def kvi(tok, c):
            return ((batch_idx * fx.Index(nkv) + kv_head_idx) * s_pad_v + tok) * HEAD_DIM + c

        def ogi(h, s, c):
            return (batch_idx * s_pad_v + s) * fx.Index(STRIDE_TOKEN) + h * HEAD_DIM + c

        def g16(ptr, i):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(i), elem_type=et)
            return _llvm.LoadOp(v16i8_t, gep).result

        def g8(ptr, i):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(i), elem_type=et)
            return _llvm.LoadOp(v8i8_t, gep).result

        q_packs = []
        for ks in range_constexpr(8):
            col = fx.Index(ks * 16) + klane * fx.Index(8)
            raw = g8(q_ptr, qgi(q_head, q_row_safe, col))
            q_packs.append(ArithValue(q_in_b).select(raw, c_z8i8))

        # 组内 128 线程合搬一个 tile
        ltid = tid % fx.Index(BLOCK_SIZE // 2)
        lrow = ltid // fx.Index(TPR)
        loff = (ltid % fx.Index(TPR)) * fx.Index(VEC)

        def gload_kv(tile):
            r0 = tile + lrow
            r1 = r0 + fx.Index(16)
            return (g16(k_ptr, kvi(r0, loff)), g16(k_ptr, kvi(r1, loff)),
                    g16(v_ptr, kvi(r0, loff)), g16(v_ptr, kvi(r1, loff)))

        c_z16i8 = Vec.filled(16, 0, fx.Int8).ir_value()

        def sk(kb, k0, k1, tile):
            # 越界行写零：pad 区 fp8 NaN 绝不进 GEMM（0×NaN=NaN 会毒化累加器，
            # P 掩码只挡得住有限值）
            ok0 = arith.cmpi(arith.CmpIPredicate.slt,
                             _raw(tile + lrow), _raw(seq_len_v))
            ok1 = arith.cmpi(arith.CmpIPredicate.slt,
                             _raw(tile + lrow + fx.Index(16)), _raw(seq_len_v))
            k0 = ArithValue(ok0).select(k0, c_z16i8)
            k1 = ArithValue(ok1).select(k1, c_z16i8)
            Vec(k0).store(lds, [kb + lrow * fx.Index(K_STRIDE) + loff])
            Vec(k1).store(lds, [kb + (lrow + fx.Index(16)) * fx.Index(K_STRIDE) + loff])

        def sv(vb, v0, v1, tile):
            for j in range_constexpr(VEC):
                d0 = loff + fx.Index(j)
                b0 = ArithValue(arith.cmpi(arith.CmpIPredicate.slt,
                                           _raw(tile + lrow), _raw(seq_len_v))).select(
                    Vec(v0)[j], Vec(c_z8i8)[0])
                b1 = ArithValue(arith.cmpi(arith.CmpIPredicate.slt,
                                           _raw(tile + lrow + fx.Index(16)), _raw(seq_len_v))).select(
                    Vec(v1)[j], Vec(c_z8i8)[0])
                Vec.from_elements([b0], fx.Int8).store(
                    lds, [vb + d0 * fx.Index(V_STRIDE) + lrow])
                Vec.from_elements([b1], fx.Int8).store(
                    lds, [vb + d0 * fx.Index(V_STRIDE) + lrow + fx.Index(16)])

        def cnt_ptr(ci):
            return create_llvm_ptr(
                extract_base_index(cnt_mr, address_space=3) + ci * fx.Index(4), 3)

        def soft_wait(ci, tgt):
            # Monotonic 原子读自旋：原子读不可被 LICM 提升出循环（普通读会变死
            # 循环），Monotonic 不触发 acquire 半边的 global_inv 降级（SeqCst/
            # Acquire fence+原子在自旋路径里发射 s_wait_loadcnt+global_inv 风暴，
            # 与在飞全局加载互锁死锁——2026-10-24 ISA 实锤）
            cur = fx.Int32(fllv.atomic_add(cnt_ptr(ci), fx.Int32(0),
                                           ordering=AtomicOrdering.Monotonic,
                                           syncscope="workgroup"))
            while arith.cmpi(arith.CmpIPredicate.slt, _raw(cur), _raw(tgt)):
                cur = fx.Int32(fllv.atomic_add(cnt_ptr(ci), fx.Int32(0),
                                               ordering=AtomicOrdering.Monotonic,
                                               syncscope="workgroup"))
            # keepalive：无副作用的 while 会被改写器当死代码静默删除（2026-10-24
            # MLIR 实锤），把观测到的计数值存 LDS 保活自旋链
            _memref.store(_raw(cur), ka_mr, [_raw(fx.Index(0))])

        def soft_arrive(ci):
            # leader（lane0）Release 到达：序住前面的 LDS store，无 acquire 半边
            if lane == fx.Index(0):
                fllv.atomic_add(cnt_ptr(ci), fx.Int32(1),
                                ordering=AtomicOrdering.Release, syncscope="workgroup")

        def soft_bar(ci, tgt):
            soft_arrive(ci)
            soft_wait(ci, tgt)

        for ci in range_constexpr(4):
            _memref.store(_raw(fx.Int32(0)), cnt_mr,
                          [_raw(grp_i * fx.Index(4) + fx.Index(ci))])
        gpu.barrier()

        def reduction_peer(v):
            return fx.Float32(v).shuffle_xor(fx.Int32(16), fx.Int32(WARP))

        # ---- 组参数：槽基址循环不变（只跨组差运行期常量，无悬崖）----
        kbA = fx.Index(LDS_G0) + grp_i * fx.Index(GRP_STRIDE)
        vaA = kbA + fx.Index(K_TILE)
        kbB = kbA + fx.Index(SLOT)
        vbB = kbB + fx.Index(K_TILE)
        ci_dataA = grp_i * fx.Index(4) + fx.Index(0)
        ci_doneA = grp_i * fx.Index(4) + fx.Index(1)
        ci_dataB = grp_i * fx.Index(4) + fx.Index(2)
        ci_doneB = grp_i * fx.Index(4) + fx.Index(3)
        g32 = fx.Int32(grp_i)
        total32 = (fx.Int32(seq_len) + fx.Int32(31)) // fx.Int32(32)
        myn = (total32 - g32 + fx.Int32(1)) // fx.Int32(2)
        npairs32 = (myn + fx.Int32(1)) // fx.Int32(2)

        # ---- 单个 tile 的计算：GEMM1 → seq 掩码 → softmax → pack → GEMM2 ----
        def compute_tile(kb_t, vb_t, tile_row, m, l, ovs):
            accs = [_raw(c_z8f), _raw(c_z8f)]
            for ks in range_constexpr(8):
                kcol = fx.Index(ks * 16) + klane * fx.Index(8)
                for half in range_constexpr(2):
                    kr = lane16 + fx.Index(half * 16)
                    kp = Vec.load(v8i8_t, lds,
                                  [kb_t + kr * fx.Index(K_STRIDE) + kcol])
                    accs[half] = wmma_fp8(to_v2i32(kp), to_v2i32(q_packs[ks]),
                                          accs[half])
            s_raw = []
            for half in range_constexpr(2):
                for r in range_constexpr(8):
                    el = fx.Float32(Vec(accs[half])[r])
                    # 任意 S：越界 k 行掩成 -1e30（p→0，lsum/GEMM2 零贡献；
                    # 顺带掐灭 pad 区 fp8 NaN。全越界 tile = 数学空操作）
                    krow = tile_row + fx.Index(half * 16) + klane * fx.Index(8) + fx.Index(r)
                    ok = arith.cmpi(arith.CmpIPredicate.slt, _raw(krow), _raw(seq_len_v))
                    s_raw.append(ArithValue(ok).select(el, c_msk))

            lmax = s_raw[0]
            for r in range_constexpr(15):
                lmax = _fmax(lmax, s_raw[r + 1])
            rmax = _fmax(lmax, reduction_peer(lmax))
            m_new = _fmax(m, rmax)
            corr = rocdl.exp2(ir.F32Type.get(),
                              _raw(_fmul(_fsub(m, m_new), qk_scale_log2e)))
            l_corr = _fmul(corr, l)
            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
            ovs = [_fmul(ov, corr_vec) for ov in ovs]

            neg_sm = _fsub(c_0f, _fmul(qk_scale_log2e, m_new))
            p_vals = []
            lsum = c_0f
            for r in range_constexpr(16):
                diff = fmath.fma(_raw(s_raw[r]), _raw(qk_scale_log2e), neg_sm)
                p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                p_vals.append(p)
                lsum = _fadd(lsum, p)
            p_packs = []
            for pks in range_constexpr(2):
                p_packs.append(pack_fp8(
                    [_fmul(p_vals[pks * 8 + j], c_448) for j in range(8)]))
            peer_sum = reduction_peer(lsum)
            tile_sum = _fadd(lsum, peer_sum)
            l = _fadd(l_corr, _fmul(tile_sum, c_448))
            m = m_new

            def ld_v(pks_v, dc_v):
                dpos = fx.Index(dc_v * 16) + lane16
                k0i = fx.Index(pks_v * 16) + klane * fx.Index(8)
                return Vec.load(v8i8_t, lds,
                                [vb_t + dpos * fx.Index(V_STRIDE) + k0i])
            cur_v = ld_v(0, 0)
            for pks in range_constexpr(2):
                for dc in range_constexpr(8):
                    has_next = const_expr(not (pks == 1 and dc == 7))
                    if const_expr(has_next):
                        ndc = dc + 1
                        npks = pks
                        if const_expr(ndc >= 8):
                            ndc = 0
                            npks = pks + 1
                        cur_v_n = ld_v(npks, ndc)
                    ovs[dc] = wmma_fp8(to_v2i32(cur_v), to_v2i32(p_packs[pks]),
                                       ovs[dc])
                    if const_expr(has_next):
                        cur_v = cur_v_n
            return m, l, ovs

        # ---- 序幕：pair 0 的全局 load（B 无效时也在界内：pad ≥ 64）----
        tA0 = grp_i * fx.Index(32)
        tB0 = tA0 + fx.Index(64)
        pkA0, pkA1, pvA0, pvA1 = gload_kv(tA0)
        pkB0, pkB1, pvB0, pvB1 = gload_kv(tB0)

        _loop_init = ([_raw(c_ninf), _raw(c_0f)] + [_raw(c_z8f)] * 8
                      + [pkA0, pkA1, pvA0, pvA1, pkB0, pkB1, pvB0, pvB1])
        loop_results = _loop_init
        for it_i, iargs in range(0, fx.Index(npairs32), 1, init=_loop_init):
            m = fx.Float32(iargs[0])
            l = fx.Float32(iargs[1])
            ovs = list(iargs[2:10])
            kA0 = iargs[10]; kA1 = iargs[11]; vA0 = iargs[12]; vA1 = iargs[13]
            kB0 = iargs[14]; kB1 = iargs[15]; vB0 = iargs[16]; vB1 = iargs[17]
            it = fx.Int32(it_i)
            tA = grp_i * fx.Index(32) + it_i * fx.Index(128)
            tB = tA + fx.Index(64)

            # ---- 槽 A：等上一对 compute A 释放 → 暂存 ----
            soft_wait(ci_doneA, fx.Int32(WPG) * it)
            sk(kbA, kA0, kA1, tA)
            sv(vaA, vA0, vA1, tA)
            soft_bar(ci_dataA, fx.Int32(WPG) * (it + fx.Int32(1)))

            # ---- 槽 B：同构（尾对 B 越界 → 掩码使 compute 成为空操作，
            #      store/barrier 依然均匀执行，无分支）----
            soft_wait(ci_doneB, fx.Int32(WPG) * it)
            sk(kbB, kB0, kB1, tB)
            sv(vbB, vB0, vB1, tB)
            soft_bar(ci_dataB, fx.Int32(WPG) * (it + fx.Int32(1)))

            # ---- 预取 pair it+1（索引钳到最后一对，永远在界内；值可弃）----
            nxt1 = it_i + fx.Index(1)
            lasti = fx.Index(npairs32 - fx.Int32(1))
            ge_last = arith.cmpi(arith.CmpIPredicate.sge, _raw(nxt1), _raw(lasti))
            nxtc = fx.Index(ArithValue(ge_last).select(lasti, nxt1))
            tAn = grp_i * fx.Index(32) + nxtc * fx.Index(128)
            tBn = tAn + fx.Index(64)
            nkA0, nkA1, nvA0, nvA1 = gload_kv(tAn)
            nB0_, nB1_, nvB0_, nvB1_ = gload_kv(tBn)

            # ---- compute A → B（B 的 LDS 早已就绪；预取在飞）----
            m, l, ovs = compute_tile(kbA, vaA, tA, m, l, ovs)
            soft_arrive(ci_doneA)
            m, l, ovs = compute_tile(kbB, vbB, tB, m, l, ovs)
            soft_arrive(ci_doneB)

            loop_results = yield [_raw(m), _raw(l)] + ovs + [
                nkA0, nkA1, nvA0, nvA1, nB0_, nB1_, nvB0_, nvB1_]
        m_run = fx.Float32(loop_results[0])
        l_run = fx.Float32(loop_results[1])
        o_run = loop_results[2:10]

        gpu.barrier()

        # ---- 合并：组 1 写 scratch（复用 K/V 槽区），组 0 读回并输出 ----
        pidx = wave % fx.Index(WPG)
        sbase = fx.Index(pidx * MERGE_F32_PER_WAVE)
        is_g1 = arith.cmpi(arith.CmpIPredicate.eq, _raw(grp), _raw(fx.Int32(1)))
        if is_g1:
            _memref.store(_raw(m_run), mrg, [_raw(sbase + lane16)])
            _memref.store(_raw(l_run), mrg, [_raw(sbase + fx.Index(16) + lane16)])
            for dc in range_constexpr(8):
                for j in range_constexpr(8):
                    d = fx.Index(dc * 16) + klane * fx.Index(8) + fx.Index(j)
                    _memref.store(_raw(fx.Float32(Vec(o_run[dc])[j])), mrg,
                                  [_raw(sbase + fx.Index(32) + lane16 * fx.Index(128) + d)])
        gpu.barrier()
        if is_g0:
            m_p = fx.Float32(_memref.load(mrg, [_raw(sbase + lane16)]))
            l_p = fx.Float32(_memref.load(mrg, [_raw(sbase + fx.Index(16) + lane16)]))
            m_star = _fmax(m_run, m_p)
            d0 = rocdl.exp2(ir.F32Type.get(),
                            _raw(_fmul(_fsub(m_run, m_star), qk_scale_log2e)))
            d1 = rocdl.exp2(ir.F32Type.get(),
                            _raw(_fmul(_fsub(m_p, m_star), qk_scale_log2e)))
            l_fin = _fadd(_fmul(d0, l_run), _fmul(d1, l_p))
            s_fin = _fmul(arith.divf(_raw(c_1f), _raw(l_fin), fastmath=fm), scale_v)
            if q_in_b:
                for dc in range_constexpr(8):
                    vals = []
                    for j in range_constexpr(8):
                        d = fx.Index(dc * 16) + klane * fx.Index(8) + fx.Index(j)
                        ov_p = fx.Float32(_memref.load(
                            mrg, [_raw(sbase + fx.Index(32) + lane16 * fx.Index(128) + d)]))
                        ov = _fadd(_fmul(fx.Float32(Vec(o_run[dc])[j]), d0), _fmul(ov_p, d1))
                        vals.append(_fmul(ov, s_fin))
                    o_vec = Vec.from_elements(vals, fx.Float32).ir_value()
                    o_bf = Vec(o_vec).to(fx.BFloat16).ir_value()
                    dcol = fx.Index(dc * 16) + klane * fx.Index(8)
                    gep = buffer_ops.get_element_ptr(
                        o_ptr, fx.Int64(ogi(q_head, q_row, dcol)), elem_type=T.bf16)
                    _llvm.StoreOp(o_bf, gep)

    @flyc.jit
    def launch_flash_attn_wsplit2(
        Q: fx.Pointer, K: fx.Pointer, V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        batch_size: fx.Int32, seq_len: fx.Int32, seq_pad: fx.Int32,
        SCL: fx.Pointer, stream: fx.Stream = fx.Stream(None),
    ):
        from flydsl.compiler.kernel_function import CompilationContext
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        nqt = (sl_idx + fx.Index(Q_TILE_ROWS - 1)) // fx.Index(Q_TILE_ROWS)
        grid_x = bs_idx * nqt * fx.Index(nkv)
        if const_expr(waves_per_eu is not None and int(waves_per_eu) >= 1):
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))
        flash_attn_fp8_wsplit2_kernel(
            Q, K, V, O, seq_len, seq_pad, SCL
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    launch_flash_attn_wsplit2.compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False},
    }

    def _ptr_arg(t):
        if hasattr(t, "data_ptr"):
            return flyc.from_c_void_p(fx.Uint8, t.data_ptr())
        return t

    def _launch(*args, **kwargs):
        args = list(args)
        for idx in [0, 1, 2, 3, 7]:
            args[idx] = _ptr_arg(args[idx])
        stream = kwargs.pop("stream", fx.Stream(None))
        _run_compiled(launch_flash_attn_wsplit2, *args, stream=stream)

    return _launch
