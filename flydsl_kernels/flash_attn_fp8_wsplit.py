# ARCHIVED: warp-split v1 experiment; production uses flash_attn_fp8_gfx1201.py.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 torchinfra — FA2 式组内 KV 分治（warp split）实验 kernel。
#
# docs/flydsl_gfx1201_toolchain.md 附6/附6补记。BM=64（4 query 头 × 16 行），
# workgroup 256 线程 = 8 wave32；wave w 认领第 (w%4) 个头的 16 行，组 g=w//4。
# 两组独立推进各自 KV tile 流（组 0 偶数、组 1 奇数），各维护 (m,l,O)，
# 末尾硬件 barrier 汇合后组 1 写 (m,l,O) 到 LDS，组 0 伙伴读回做 softmax 合并：
#   O = (O0·d0 + O1·d1)/(l0·d0 + l1·d1)·sv,  d_g = exp2((m_g-m*)·scale)
#
# ── 组内软 barrier 最终配方（2026-10-24 定型，全链路 MLIR+ISA 实证）──
# 1. 组循环必须是 for+yield 携带、不能被 if/else 分支包裹（改写器在分支内
#    包 while 会静默丢弃嵌套 while——旧版 4 个自旋全没编译进去，纯竞态 NaN）。
# 2. 自旋 while 体内必须保活（结束时把观测值存 LDS ka 槽）：无副作用的
#    while 会被改写器当死代码删除。
# 3. 自旋读必须用 Monotonic 原子读（atomic_add +0）：普通读会被 LICM 提出
#    循环变成死循环；Acquire/SeqCst 会在每次 poll 发射 s_wait_loadcnt+
#    global_inv(SE)，与在飞全局加载互锁死锁。
# 4. 到达用 lane0 Release 原子（序住 LDS store，无 acquire 半边）；全程
#    零 memory_fence。
#
# LDS：cnt[4]@0 | g0 K@16 V@4240 | g1 K@8848 V@13072 | merge@17680 | ka@50960
#   每伙伴 wave 槽 = 32(m,l) + 2048(O) f32 = 8320B，共 4 槽 → ~50.9KB。
#
# 首版性能（vs legacy 同条件）：59.6/74.2/70.3 TF @2K/8K/32K
# （legacy 140/99.7/93.4）。已知差距：K/V 单缓冲、V 转置存储是每线程
# 32×1B 散射、leader 到达、无 wpe 调优。双缓冲/全 lane 到达/V 向量化
# 是后续优化项。正确性：S≤2048 vs fp32 参考 Δ≤0.0032；8K/32K vs
# legacy Δ≤0.0006。

import math as host_math

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
LDS_G0 = 16
LDS_G1 = LDS_G0 + K_TILE + V_TILE
LDS_MERGE = LDS_G1 + K_TILE + V_TILE
MERGE_F32_PER_WAVE = 32 + 2048
MERGE_BYTES = MERGE_F32_PER_WAVE * 4 * 4
KA_OFF = LDS_MERGE + MERGE_BYTES          # keepalive 槽：防 DCE 丢自旋
LDS_TOTAL = KA_OFF + 4


def build_flash_attn_fp8_wsplit_module(
    num_heads,
    head_dim,
    num_kv_heads=None,
    waves_per_eu=None,
    flat_work_group_size=WAVES * WARP,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    dbg=False,
):
    assert head_dim == 128
    nkv = num_kv_heads if num_kv_heads is not None else num_heads
    assert num_heads == 4 * nkv, "BM=64 需要 NQ/NKV=4（Nemotron 32/8）"
    NQ_PER_KV = 4
    HEAD_DIM = head_dim
    STRIDE_TOKEN = num_heads * HEAD_DIM
    BLOCK_SIZE = flat_work_group_size
    VEC = 16
    TPR = HEAD_DIM // VEC
    FAITHFUL = False  # wsplit 用 cheap l（×448 补偿，主 kernel 已验证同精度）

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="wsplit_smem")
    _off = allocator._align(allocator.ptr, 16)
    allocator.ptr = _off + LDS_TOTAL   # 不推进 = LDS 分配 0 字节（全越界）

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_fp8_wsplit_kernel(
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
        cnt_mr = SmemPtr(allocator.get_base(), 0, T.i32, shape=(4,)).get()
        ka_mr = SmemPtr(allocator.get_base(), KA_OFF, T.i32, shape=(1,)).get()
        mrg = SmemPtr(allocator.get_base(), LDS_MERGE, T.f32,
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
        num_q_tiles = (seq_len_v + fx.Index(15)) // fx.Index(16)
        q_tile_idx = bqt % num_q_tiles
        batch_idx = bqt // num_q_tiles
        q_start = q_tile_idx * fx.Index(16)

        q_head = fx.Index(kv_head_idx * NQ_PER_KV) + (wave % fx.Index(WPG))
        q_row = q_start + lane16
        q_in_b = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        q_row_safe = fx.Index(ArithValue(q_in_b).select(q_row, fx.Index(0)))
        c_z8i8 = Vec.filled(8, 0, fx.Int8).ir_value()
        c_z8f = Vec.filled(8, 0.0, fx.Float32)
        c_ninf = fx.Float32(float("-inf"))
        c_0f = fx.Float32(0.0)
        c_1f = fx.Float32(1.0)
        c_448 = fx.Float32(448.0)

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

        def sk(kb, k0, k1):
            Vec(k0).store(lds, [kb + lrow * fx.Index(K_STRIDE) + loff])
            Vec(k1).store(lds, [kb + (lrow + fx.Index(16)) * fx.Index(K_STRIDE) + loff])

        def sv(vb, v0, v1):
            for j in range_constexpr(VEC):
                d0 = loff + fx.Index(j)
                Vec.from_elements([Vec(v0)[j]], fx.Int8).store(
                    lds, [vb + d0 * fx.Index(V_STRIDE) + lrow])
                Vec.from_elements([Vec(v1)[j]], fx.Int8).store(
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

        _memref.store(_raw(fx.Int32(0)), cnt_mr, [_raw(grp_i)])
        _memref.store(_raw(fx.Int32(0)), cnt_mr, [_raw(fx.Index(2) + grp_i)])
        gpu.barrier()

        def reduction_peer(v):
            return fx.Float32(v).shuffle_xor(fx.Int32(16), fx.Int32(WARP))

        # 单路径组循环：组参数运行时算术（grp_i），外层 for + 携带。
        # 不用 if/else 包 while：改写器会静默丢弃 while 嵌 while 的内层自旋
        # （2026-10-24 MLIR 实锤：wsplit 旧版 4 个自旋全丢 → 纯竞态 NaN）。
        kb = fx.Index(LDS_G0) + grp_i * fx.Index(K_TILE + V_TILE)
        vb = kb + fx.Index(K_TILE)
        ci_data = grp_i
        ci_done = fx.Index(2) + grp_i
        g32 = fx.Int32(grp_i)
        total32 = (fx.Int32(seq_len) + fx.Int32(31)) // fx.Int32(32)
        myn = (total32 - g32 + fx.Int32(1)) // fx.Int32(2)

        _loop_init = [_raw(c_ninf), _raw(c_0f)] + [_raw(c_z8f) for _ in range(8)]
        loop_results = _loop_init
        for it_i, iargs in range(0, fx.Index(myn), 1, init=_loop_init):
            m = fx.Float32(iargs[0])
            l = fx.Float32(iargs[1])
            o0_ = iargs[2]; o1_ = iargs[3]; o2_ = iargs[4]; o3_ = iargs[5]
            o4_ = iargs[6]; o5_ = iargs[7]; o6_ = iargs[8]; o7_ = iargs[9]
            it = fx.Int32(it_i)
            tile = grp_i * fx.Index(32) + it_i * fx.Index(64)
            k0, k1, v0, v1 = gload_kv(tile)
            soft_wait(ci_done, fx.Int32(WPG) * it)
            sk(kb, k0, k1)
            sv(vb, v0, v1)
            soft_bar(ci_data, fx.Int32(WPG) * (it + fx.Int32(1)))

            # ---- GEMM1 ----
            accs = [_raw(c_z8f), _raw(c_z8f)]
            for ks in range_constexpr(8):
                kcol = fx.Index(ks * 16) + klane * fx.Index(8)
                for half in range_constexpr(2):
                    kr = lane16 + fx.Index(half * 16)
                    kp = Vec.load(v8i8_t, lds,
                                  [kb + kr * fx.Index(K_STRIDE) + kcol])
                    accs[half] = wmma_fp8(to_v2i32(kp), to_v2i32(q_packs[ks]), accs[half])
            s_raw = []
            for half in range_constexpr(2):
                for r in range_constexpr(8):
                    s_raw.append(Vec(accs[half])[r])

            # ---- softmax 标量链 ----
            lmax = s_raw[0]
            for r in range_constexpr(15):
                lmax = _fmax(lmax, s_raw[r + 1])
            rmax = _fmax(lmax, reduction_peer(lmax))
            m_new = _fmax(m, rmax)
            corr = rocdl.exp2(ir.F32Type.get(),
                              _raw(_fmul(_fsub(m, m_new), qk_scale_log2e)))
            l_corr = _fmul(corr, l)
            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
            o0_ = _fmul(o0_, corr_vec); o1_ = _fmul(o1_, corr_vec)
            o2_ = _fmul(o2_, corr_vec); o3_ = _fmul(o3_, corr_vec)
            o4_ = _fmul(o4_, corr_vec); o5_ = _fmul(o5_, corr_vec)
            o6_ = _fmul(o6_, corr_vec); o7_ = _fmul(o7_, corr_vec)

            # ---- exp2 + pack ----
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

            # ---- GEMM2（V^T LDS，A 操作数预取流水）----
            def ld_v(pks_v, dc_v):
                dpos = fx.Index(dc_v * 16) + lane16
                k0i = fx.Index(pks_v * 16) + klane * fx.Index(8)
                return Vec.load(v8i8_t, lds,
                                [vb + dpos * fx.Index(V_STRIDE) + k0i])
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
                    accs_l = [o0_, o1_, o2_, o3_, o4_, o5_, o6_, o7_]
                    accs_l[dc] = wmma_fp8(to_v2i32(cur_v), to_v2i32(p_packs[pks]), accs_l[dc])
                    o0_, o1_, o2_, o3_ = accs_l[0], accs_l[1], accs_l[2], accs_l[3]
                    o4_, o5_, o6_, o7_ = accs_l[4], accs_l[5], accs_l[6], accs_l[7]
                    if const_expr(has_next):
                        cur_v = cur_v_n
            soft_arrive(ci_done)
            loop_results = yield [_raw(m), _raw(l),
                                  o0_, o1_, o2_, o3_, o4_, o5_, o6_, o7_]
        m_run = fx.Float32(loop_results[0])
        l_run = fx.Float32(loop_results[1])
        o_run = loop_results[2:10]

        gpu.barrier()

        # ---- 合并：组 1 写 scratch，组 0 读回并输出 ----
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
            if const_expr(dbg):
                inv_ld = arith.divf(_raw(c_1f), _raw(l_run), fastmath=fm)
                s_solo = _fmul(inv_ld, scale_v)
                if q_in_b:
                    for dc in range_constexpr(8):
                        vals = [_fmul(fx.Float32(Vec(o_run[dc])[j]), s_solo) for j in range(8)]
                        o_vec2 = Vec.from_elements(vals, fx.Float32).ir_value()
                        o_bf2 = Vec(o_vec2).to(fx.BFloat16).ir_value()
                        dcol2 = fx.Index(dc * 16) + klane * fx.Index(8)
                        gep2 = buffer_ops.get_element_ptr(
                            o_ptr, fx.Int64(ogi(q_head, q_row, dcol2)), elem_type=T.bf16)
                        _llvm.StoreOp(o_bf2, gep2)
            m_p = fx.Float32(_memref.load(mrg, [_raw(sbase + lane16)]))
            l_p = fx.Float32(_memref.load(mrg, [_raw(sbase + fx.Index(16) + lane16)]))
            m_star = _fmax(m_run, m_p)
            d0 = rocdl.exp2(ir.F32Type.get(),
                            _raw(_fmul(_fsub(m_run, m_star), qk_scale_log2e)))
            d1 = rocdl.exp2(ir.F32Type.get(),
                            _raw(_fmul(_fsub(m_p, m_star), qk_scale_log2e)))
            l_fin = _fadd(_fmul(d0, l_run), _fmul(d1, l_p))
            if const_expr(dbg):
                import struct as _st
                dbg_vals = [_fmul(m_run, fx.Float32(0.001)), _fmul(l_run, fx.Float32(0.001)),
                            _fmul(m_p, fx.Float32(0.001)), _fmul(l_p, fx.Float32(0.001))]
                dbgv = Vec.from_elements(dbg_vals, fx.Float32).ir_value()
                dbgb = Vec(dbgv).to(fx.BFloat16).ir_value()
                dgp = buffer_ops.get_element_ptr(
                    o_ptr, fx.Int64(ogi(q_head, q_row, fx.Index(0))), elem_type=T.bf16)
                _llvm.StoreOp(_raw(fx.Float32(dbg_vals[0])), dgp) if False else None
                # 4 个 bf16 标量存（诊断专用）
                for _di in range_constexpr(4):
                    _gp = buffer_ops.get_element_ptr(
                        o_ptr, fx.Int64(ogi(q_head, q_row, fx.Index(_di))), elem_type=T.bf16)
                    _llvm.StoreOp(_raw(Vec.from_elements([dbg_vals[_di]], fx.Float32).to(fx.BFloat16)), _gp)
            s_fin = _fmul(arith.divf(_raw(c_1f), _raw(l_fin), fastmath=fm), scale_v)
            if const_expr(not dbg) and q_in_b:
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
    def launch_flash_attn_wsplit(
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
        nqt = (sl_idx + fx.Index(15)) // fx.Index(16)
        grid_x = bs_idx * nqt * fx.Index(nkv)
        if const_expr(waves_per_eu is not None and int(waves_per_eu) >= 1):
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))
        flash_attn_fp8_wsplit_kernel(
            Q, K, V, O, seq_len, seq_pad, SCL
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    launch_flash_attn_wsplit.compile_hints = {
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
        _run_compiled(launch_flash_attn_wsplit, *args, stream=stream)

    return _launch
