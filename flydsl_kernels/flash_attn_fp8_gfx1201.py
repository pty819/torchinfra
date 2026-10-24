# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors.
# Copyright (c) 2026 torchinfra — fp8 adaptation.
#
# FP8 Flash Attention forward for gfx1201, derived from aiter's
# flash_attn_func_gfx1201.py (bf16). Structure is preserved 1:1 (BM=128/BN=32,
# 8 waves, Q resident in registers, K/V via padded LDS, V prefetch pipeline);
# the GEMMs switch to wmma_f32_16x16x16_fp8_fp8 (v2i32 fragments, 8B/lane).
#
# FP8 math follows torchinfra P14 (validated E2E there):
#   - Q/K/V arrive pre-quantized e4m3 with tensorwise scales (host folds
#     scale_q*scale_k*sm_scale*log2e into ONE runtime arg, scale_v another);
#   - online softmax stays fp32;
#   - P is row-max-normalized (<=1), quantized to fp8 with *448 (FA3 style,
#     no clipping), PV runs fp8 x fp8;
#   - the softmax denominator sums the QUANTIZED P (dequantized) so numerator
#     and denominator live in the same quantization domain and the 448 factor
#     cancels in O = (P8@V8)/l_q * scale_v.

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from .kernels_common import dtype_to_elem_type
from .tensor_shim import _run_compiled
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    llvm as _llvm,
    memref as _memref,
)

KERNEL_NAME = "flash_attn_fp8_gfx1201_kernel"
_LOG2E = host_math.log2(host_math.e)


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _pointer_to_llvm_ptr(ptr) -> ir.Value:
    ptr_i64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), ptr_i64).result


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    _llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def build_flash_attn_fp8_module(
    num_heads,
    head_dim,
    num_kv_heads=None,
    causal=True,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
    v_layout="transposed",  # 'transposed': V^T in LDS ([D][BN]) -> GEMM2 连续 8B 向量读
    #   'row': aiter bf16 原版 [BN][D] -> GEMM2 每 lane 8 次 1B 标量读（诊断基线）
    ablate="none",  # 'no_pv': 跳过 GEMM2（诊断 QK+softmax 成本，输出错误）
    #   'synth_p': 跳过 P 构造链（exp2/fma/cvt/l 打包），GEMM2 喂循环外预打包的
    #   常数片段（输出错误；差值 = P 构造链的真实成本，含依赖链缩短效应）
    #   'neither': 两者都跳（纯 GEMM1+softmax+store 地板）
    faithful_l=True,  # False: 分母用 exp2 域求和（省 8 cvt + 16 adds/tile）
    kv_merge=1,  # 2: 跨 tile softmax 合并——两个 KV tile 一次 m/l/corr 更新，
    #   GEMM1 背靠背独立发射，exp2/pack 链 ILP x2（打串行依赖链，非吞吐瓶颈）
    #   实测持平：GEMM1/exp2 减半被 GEMM2 C-累加链翻倍抵消
    sw_pipeline=False,  # True: K/V 双 buffer LDS + 寄存器跨 barrier 暂存，
    #   GEMM2[t] 与 GEMM1[t+1] 之间无 barrier 可重叠（打掉三段串行结构）
):
    """Build gfx1201 fp8 flash attention (e4m3 Q/K/V in, bf16 O out)."""
    gpu_arch = get_hip_arch()

    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    K_SUB_N = 32
    ROWS_PER_WAVE = WMMA_M

    BLOCK_M = block_m if block_m is not None else 128
    BLOCK_N = block_n if block_n is not None else 32

    assert BLOCK_N % K_SUB_N == 0
    assert BLOCK_M % ROWS_PER_WAVE == 0

    N_SUB_TILES = BLOCK_N // K_SUB_N
    KV_MERGE = int(kv_merge)
    assert KV_MERGE in (1, 2)
    assert KV_MERGE == 1 or not causal, "kv_merge=2 只支持非 causal（rerank 双向）"
    SW_PIPE = bool(sw_pipeline)
    assert not SW_PIPE or (KV_MERGE == 1 and not causal), "sw_pipeline 只支持 merge=1 非 causal"
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8
    NUM_S_VALS_M = NUM_S_VALS * KV_MERGE  # 合并后每 lane 的 S 值总数

    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    FAITHFUL_L = faithful_l
    PATH_TAG = f"M{BLOCK_M}N{BLOCK_N}_fp8"
    BLOCK_N_OUT = BLOCK_N

    NUM_PREFETCH_K = 1
    NUM_PREFETCH_V = 1

    K_STEP_QK = WMMA_K
    K_STEPS_QK = head_dim // K_STEP_QK  # 128/16 = 8
    WMMA_LANE_K = 8

    D_CHUNK = WMMA_N
    D_CHUNKS = head_dim // D_CHUNK  # 8

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 32/16 = 2

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0
    assert head_dim >= 64

    NUM_HEADS = num_heads
    NUM_KV_HEADS = num_kv_heads if num_kv_heads is not None else num_heads
    NQ_PER_KV = NUM_HEADS // NUM_KV_HEADS
    assert NUM_HEADS % NUM_KV_HEADS == 0
    # GQA row mapping: BM rows = NQ_PER_KV query heads x ROWS_PER_HEAD
    # positions of the shared KV head. Waves are 16 rows; ROWS_PER_HEAD must
    # be a multiple of 16 so a wave never straddles two query heads.
    ROWS_PER_HEAD = BLOCK_M // NQ_PER_KV
    assert ROWS_PER_HEAD >= 16 and ROWS_PER_HEAD % 16 == 0, (
        f"BLOCK_M={BLOCK_M} / NQ_PER_KV={NQ_PER_KV} must give >=16 and 16 | rows/head"
    )
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM  # output BSHD row stride (elements)

    # LDS layout: element = 1 byte for fp8; pad by bytes
    K_STRIDE = HEAD_DIM + 4
    _vt_pad = int(os.getenv("FP8_VT_PAD", "4"))
    if v_layout == "transposed":
        V_STRIDE = BLOCK_N + _vt_pad  # [D][BN] transposed tile
    else:
        V_STRIDE = HEAD_DIM + 4  # [BN][D] row-major

    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    # transposed V tile is [D][BN]: D rows of BLOCK_N columns (+pad)
    LDS_V_TILE_SIZE = (HEAD_DIM if v_layout == "transposed" else BLOCK_N) * V_STRIDE
    # kv_merge=2: K/V 各双 buffer（两个 tile 的装载共一个 barrier，GEMM1 背靠背）
    # sw_pipeline: K/V 各三 buffer——barrier 只保证波间偏差 ≤1 迭代，mod 3 才能
    # 保证 stores[t+1] 不撞 GEMM2[t] 正在读的 buffer（mod 2 有竞态）
    _KV_BUFS = 3 if SW_PIPE else (2 if KV_MERGE > 1 else 1)
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * _KV_BUFS * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * _KV_BUFS * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_fp8_gfx1201_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    # fp8 elements: 1 byte each (bf16 kernel multiplied by 2 for byte sizing)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE

    elem_out = fx.BFloat16  # output tensor element (numeric class, context-free)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def _kernel_pipe(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,   # real S: kv loop bound, q in-bounds, store bound
        seq_pad: fx.Int32,   # padded S (=tensor layout stride); pad rows never read
        SCL: fx.Pointer,  # device [2] f32: qk_scale_log2e, scale_v (zero host sync)
    ):
        elem_type = T.i8  # LDS & input pointer element (fp8 bytes) — needs MLIR ctx
        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        o_ptr = _pointer_to_llvm_ptr(O)
        scl_ptr = _pointer_to_llvm_ptr(SCL)
        qk_scale_log2e = fx.Float32(
            _pointer_load(ir.F32Type.get(),
                          buffer_ops.get_element_ptr(scl_ptr, fx.Int64(0), elem_type=T.f32))
        )
        scale_v = fx.Float32(
            _pointer_load(ir.F32Type.get(),
                          buffer_ops.get_element_ptr(scl_ptr, fx.Int64(1), elem_type=T.f32))
        )
        fm_fast = arith.FastMathFlags.fast

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        v8f32_type = Vec.make_type(8, fx.Float32)
        v8i8_type = Vec.make_type(8, fx.Int8)
        vxf8_type = Vec.make_type(VEC_WIDTH, fx.Int8)
        v2f32_type = Vec.make_type(2, fx.Float32)

        def to_v2i32(x):
            return Vec(x).bitcast(fx.Int32)

        def wmma_fp8(a_v2i32, b_v2i32, c_v8f32):
            return rocdl.wmma_f32_16x16x16_fp8_fp8(
                v8f32_type, _raw(a_v2i32), _raw(b_v2i32), c_v8f32
            ).result

        seq_len_v = fx.Index(seq_len)

        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        wave_q_offset = wave_id * ROWS_PER_WAVE

        s_pad_v = fx.Index(seq_pad)
        kv_head_idx = block_id % NUM_KV_HEADS
        batch_q_tile_id = block_id // NUM_KV_HEADS
        num_q_tiles = (seq_len_v + ROWS_PER_HEAD - 1) // ROWS_PER_HEAD
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * ROWS_PER_HEAD

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def kv_global_idx(token_idx, col):
            # K/V: [B, NKV, S, D] BHSD contiguous (fp8 bytes)
            return ((batch_idx * NUM_KV_HEADS + kv_head_idx) * s_pad_v + token_idx) * HEAD_DIM + col

        def q_global_idx(q_head, q_s, col):
            # Q: [B, NQ, S, D] BHSD contiguous (fp8 bytes)
            return ((batch_idx * NUM_HEADS + q_head) * s_pad_v + q_s) * HEAD_DIM + col

        def out_global_idx(q_head, q_s, col):
            # O: [B, S, NQ*D] bf16, token-major (matches torchinfra's layout)
            return (batch_idx * s_pad_v + q_s) * STRIDE_TOKEN + q_head * HEAD_DIM + col

        def _load_global_vec(ptr, base_idx, vec_type):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(vec_type, gep)

        def _store_global_out(ptr, base_idx, val):
            # O is bf16: element indices are the same numbers but the GEP
            # element type (and thus the byte stride) is bf16
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.bf16)
            _pointer_store(val, gep)

        def _pack_fp8_pair(f_lo, f_hi, old_dword, word_sel):
            return rocdl.cvt_pk_fp8_f32(
                T.i32, _raw(fx.Float32(f_lo)), _raw(fx.Float32(f_hi)), _raw(old_dword), word_sel
            )

        def fp8_pack_v8_(f32_vals):
            c_zero_i32 = fx.Int32(0)
            d0 = _pack_fp8_pair(f32_vals[0], f32_vals[1], c_zero_i32, 0)
            d0 = _pack_fp8_pair(f32_vals[2], f32_vals[3], d0, 1)
            d1 = _pack_fp8_pair(f32_vals[4], f32_vals[5], c_zero_i32, 0)
            d1 = _pack_fp8_pair(f32_vals[6], f32_vals[7], d1, 1)
            return Vec.from_elements([d0, d1], fx.Int32).ir_value()

        def fp8_unpack_sum(dword):
            """Sum the 4 fp8 values inside one i32 dword (faithful l)."""
            va = rocdl.cvt_pk_f32_fp8(v2f32_type, _raw(dword), 0)
            vb = rocdl.cvt_pk_f32_fp8(v2f32_type, _raw(dword), 1)
            s = _fadd(Vec(va)[0], Vec(va)[1])
            s = _fadd(s, Vec(vb)[0])
            s = _fadd(s, Vec(vb)[1])
            return s

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)
            return fx.Index(LDS_V_BASE) + buf_id * fx.Index(LDS_V_TILE_SIZE)

        def coop_load_k_global(tile_start):
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = kv_global_idx(row_idx, load_col_base)
                vecs.append(_load_global_vec(k_ptr, g_idx, vxf8_type))
            return vecs

        def coop_store_k_lds(vecs, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                lds_idx = k_base + (load_row_in_batch + row_offset) * K_STRIDE + load_col_base
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        Vec(vecs[batch]).store(lds_kv, [lds_idx])
                else:
                    Vec(vecs[batch]).store(lds_kv, [lds_idx])

        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = kv_global_idx(row_idx, load_col_base)
                        lds_idx = k_base + (load_row_in_batch + row_offset) * K_STRIDE + load_col_base
                        vec = _load_global_vec(k_ptr, g_idx, vxf8_type)
                        Vec(vec).store(lds_kv, [lds_idx])
                else:
                    g_idx = kv_global_idx(row_idx, load_col_base)
                    lds_idx = k_base + (load_row_in_batch + row_offset) * K_STRIDE + load_col_base
                    vec = _load_global_vec(k_ptr, g_idx, vxf8_type)
                    Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_transposed(v_base, lds_row, vec):
            # scatter VEC_WIDTH bytes of one KV row into [D][BN] layout:
            # 16 scalar 1B stores, but GEMM2 then reads 8B vectors
            for j in range_constexpr(VEC_WIDTH):
                d_col = load_col_base + j
                byte_idx = v_base + d_col * V_STRIDE + lds_row
                Vec.from_elements([Vec(vec)[j]], fx.Int8).store(lds_kv, [byte_idx])

        def coop_load_v_global(tile_start):
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = kv_global_idx(row_idx, load_col_base)
                vecs.append(_load_global_vec(v_ptr, g_idx, vxf8_type))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                store_v = _v_store_transposed if v_layout == "transposed" else _v_store_row_major
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        store_v(v_base, load_row_in_batch + row_offset, vecs[batch])
                else:
                    store_v(v_base, load_row_in_batch + row_offset, vecs[batch])

        # ---- Q preload (8 fp8 bytes per kstep per lane) ----
        # GQA: this wave's 16 rows live inside ONE query head (16 | ROWS_PER_HEAD)
        # GQA row mapping: block row r -> (head_local = r//ROWS_PER_HEAD,
        # head-local seq = q_start + r%ROWS_PER_HEAD); wave never straddles
        # heads, so the div/mod are wave-uniform
        q_row = q_start + (wave_q_offset % fx.Index(ROWS_PER_HEAD)) + lane16
        q_head_local = wave_q_offset // fx.Index(ROWS_PER_HEAD)  # 0..NQ_PER_KV-1
        q_head = fx.Index(kv_head_idx * NQ_PER_KV) + q_head_local
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        c_zero_v8i8 = Vec.filled(8, 0, fx.Int8).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = q_global_idx(q_head, q_row_safe, q_col)
            raw = _load_global_vec(q_ptr, g_idx, v8i8_type)
            q_b_packs.append(ArithValue(q_in_bounds).select(raw, c_zero_v8i8))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_log2_pmax = fx.Float32(host_math.log2(448.0))  # fold into exp2 bias
        c_pmax = fx.Float32(448.0)  # e4m3 max; p<=1 so *448 never clips
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        # synth_p 诊断：循环外预打包的常数 P 片段（448.0 的 e4m3 编码），
        # 行数按 kv_merge 后的合并子 tile 数
        synth_p_packs = []
        if const_expr(ablate in ("synth_p", "neither")):
            for _st in range_constexpr(N_SUB_TILES * KV_MERGE):
                _row = []
                for _pks in range_constexpr(PV_K_STEPS):
                    _row.append(fp8_pack_v8_([c_pmax] * 8))
                synth_p_packs.append(_row)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)

        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_16_i32, width_i32)

        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v))
        else:
            kv_upper = seq_len_v
        # ==== 软件流水结构（merge=1 专用） ====
        # iter t: GEMM1[t](K buf p) -> softmax/exp2/pack[t] -> 存 K/V[t+1] 到
        # buf p+1 -> barrier -> GEMM2[t](V buf p) -> 预取 K/V[t+2]。
        # GEMM2[t] 与下迭代 GEMM1[t+1] 之间无 barrier：装载 / exp2-pack /
        # GEMM2 三段原本各 ~3.9ms 全串行，流水后互相重叠。
        def _gemm1(k_base):
            accs = [_raw(c_zero_v8f32) for _ in range(NUM_S_ACCS)]
            for ks in range_constexpr(K_STEPS_QK):
                k_col = fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N
                    k_row_a = lane16 + fx.Index(st_base_row)
                    k_lds_a = k_base + k_row_a * K_STRIDE + k_col
                    k_pack_a = Vec.load(v8i8_type, lds_kv, [k_lds_a])
                    k_row_b = lane16 + fx.Index(st_base_row + 16)
                    k_lds_b = k_base + k_row_b * K_STRIDE + k_col
                    k_pack_b = Vec.load(v8i8_type, lds_kv, [k_lds_b])
                    accs[st_idx * 2] = wmma_fp8(
                        to_v2i32(k_pack_a), to_v2i32(q_b_packs[ks]), accs[st_idx * 2]
                    )
                    accs[st_idx * 2 + 1] = wmma_fp8(
                        to_v2i32(k_pack_b), to_v2i32(q_b_packs[ks]), accs[st_idx * 2 + 1]
                    )
            return accs

        def _load_v_pipe(v_base, st_kv_base_val, pks_val, dc_val):
            if const_expr(v_layout == "transposed"):
                d_pos = fx.Index(dc_val * D_CHUNK) + lane16
                k0 = fx.Index(st_kv_base_val + pks_val * PV_K_STEP) + klane * WMMA_LANE_K
                v_lds_idx = v_base + d_pos * V_STRIDE + k0
                return Vec.load(v8i8_type, lds_kv, [v_lds_idx])
            d_pos = fx.Index(dc_val * D_CHUNK) + lane16
            v_elems = []
            for k_sub in range_constexpr(8):
                kv_row = (
                    fx.Index(st_kv_base_val + pks_val * PV_K_STEP)
                    + klane * WMMA_LANE_K
                    + fx.Index(k_sub)
                )
                v_elems.append(_memref.load(lds_kv, [_raw(v_base + kv_row * V_STRIDE + d_pos)]))
            return Vec.from_elements(v_elems, fx.Int8).ir_value()

        _k0 = coop_load_k_global(fx.Index(0))
        coop_store_k_lds(_k0, 0)
        _v0 = coop_load_v_global(fx.Index(0))
        coop_store_v_lds(_v0, 0)
        gpu.barrier()
        _k_next = coop_load_k_global(fx.Index(0))
        _v_next = coop_load_v_global(fx.Index(0))

        _pipe_init = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            _pipe_init.append(_raw(c_zero_v8f32))
        for batch in range_constexpr(NUM_BATCHES_KV):
            _pipe_init.append(_k_next[batch])
        for batch in range_constexpr(NUM_BATCHES_KV):
            _pipe_init.append(_v_next[batch])

        loop_results = _pipe_init
        for kv_block_start, inner_iter_args in range(
            0, kv_upper, BLOCK_N_OUT, init=_pipe_init
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _k_vecs = [
                inner_iter_args[2 + D_CHUNKS + b]
                for b in range_constexpr(NUM_BATCHES_KV)
            ]
            _v_vecs_prefetch = [
                inner_iter_args[2 + D_CHUNKS + NUM_BATCHES_KV + b]
                for b in range_constexpr(NUM_BATCHES_KV)
            ]
            _tile_id = kv_block_start // fx.Index(BLOCK_N_OUT)
            _buf = _tile_id % fx.Index(3)

            # ==== staged 实验：迭代开头存本 tile 的 K/V 再 barrier ====
            coop_store_k_lds(_k_vecs, _buf)
            coop_store_v_lds(_v_vecs_prefetch, _buf)
            gpu.barrier()

            # ==== GEMM1: S = K @ Q^T (fp8 x fp8) ====
            s_accs = _gemm1(k_buf_base(_buf))

            # ==== Online softmax (fp32) ====
            s_raw = []
            for st in range_constexpr(NUM_S_ACCS):
                for r in range_constexpr(8):
                    s_raw.append(Vec(s_accs[st])[r])

            local_max = s_raw[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max = _fmax(local_max, s_raw[r + 1])
            peer_max = reduction_peer(local_max)
            row_max = _fmax(local_max, peer_max)
            m_new_raw = _fmax(m_running, row_max)

            diff_m_raw = _fsub(m_running, m_new_raw)
            diff_m_scaled = _fmul(diff_m_raw, qk_scale_log2e)
            corr = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_scaled))
            l_corr = _fmul(corr, l_running)

            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

            if const_expr(ablate in ("synth_p", "neither")):
                l_new = l_corr
                p_packs_all = synth_p_packs
            else:
                scaled_max = _fmul(qk_scale_log2e, m_new_raw)
                neg_scaled_max = _fsub(c_zero_f, scaled_max)
                p_vals = []
                local_sum = _raw(c_zero_f)
                for r in range_constexpr(NUM_S_VALS):
                    diff = fmath.fma(_raw(s_raw[r]), _raw(qk_scale_log2e), neg_scaled_max)
                    p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                    p_vals.append(p)
                    local_sum = _fadd(local_sum, p)
                peer_sum = reduction_peer(local_sum)
                tile_sum = _fadd(local_sum, peer_sum)

                # ==== Build P packs: p*448 -> e4m3（barrier 前算好，GEMM2 释放即发射）====
                p_packs_all = []
                p_dwords = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    p_packs_st = []
                    for pks in range_constexpr(PV_K_STEPS):
                        acc_idx = st_idx * 2 + pks
                        p_base = acc_idx * 8
                        p_slice = [p_vals[p_base + j] for j in range(8)]
                        scaled = [_fmul(p_slice[j], c_pmax) for j in range(8)]
                        pack = fp8_pack_v8_(scaled)
                        p_packs_st.append(pack)
                        p_dwords.append(pack)
                    p_packs_all.append(p_packs_st)

                if const_expr(FAITHFUL_L):
                    local_lq = _raw(c_zero_f)
                    for d_idx in range_constexpr(N_SUB_TILES * PV_K_STEPS):
                        local_lq = _fadd(local_lq, fp8_unpack_sum(Vec(p_dwords[d_idx])[0]))
                        local_lq = _fadd(local_lq, fp8_unpack_sum(Vec(p_dwords[d_idx])[1]))
                    peer_lq = reduction_peer(local_lq)
                    tile_lq = _fadd(local_lq, peer_lq)
                    l_new = _fadd(l_corr, tile_lq)
                else:
                    l_new = _fadd(l_corr, _fmul(tile_sum, c_pmax))

            # ==== GEMM2: O += V^T @ P（V buf p；与下迭代 GEMM1 之间无 barrier）====
            v_base = v_buf_base(_buf)
            cur_v_packs = []
            for st_idx in range_constexpr(N_SUB_TILES):
                cur_v_packs.append(_load_v_pipe(v_base, st_idx * K_SUB_N, 0, 0))

            for pks in range_constexpr(PV_K_STEPS):
                for dc in range_constexpr(D_CHUNKS):
                    next_dc = dc + 1
                    next_pks = pks
                    if const_expr(next_dc >= D_CHUNKS):
                        next_dc = 0
                        next_pks = pks + 1
                    has_next = const_expr(next_pks < PV_K_STEPS)
                    next_v_packs = []
                    if const_expr(has_next):
                        for st_idx in range_constexpr(N_SUB_TILES):
                            next_v_packs.append(
                                _load_v_pipe(v_base, st_idx * K_SUB_N, next_pks, next_dc)
                            )
                    for st_idx in range_constexpr(N_SUB_TILES):
                        if const_expr(ablate not in ("no_pv", "neither")):
                            o_accs[dc] = wmma_fp8(
                                to_v2i32(cur_v_packs[st_idx]),
                                to_v2i32(p_packs_all[st_idx][pks]),
                                o_accs[dc],
                            )
                    if const_expr(has_next):
                        cur_v_packs = next_v_packs

            m_running = m_new_raw
            l_running = l_new

            next_kv_start = kv_block_start + fx.Index(BLOCK_N_OUT)
            _k_load = coop_load_k_global(next_kv_start)
            _v_load = coop_load_v_global(next_kv_start)
            _yield_args = [m_running, l_running] + o_accs
            for batch in range_constexpr(NUM_BATCHES_KV):
                _yield_args.append(_k_load[batch])
            for batch in range_constexpr(NUM_BATCHES_KV):
                _yield_args.append(_v_load[batch])
            loop_results = yield _yield_args


        # ---- Normalize and store O (bf16 out, descale by scale_v) ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()
        sv_vec = Vec.from_elements([scale_v], fx.Float32).broadcast_to(8).ir_value()

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = _fmul(o_finals[dc], inv_l_vec)
                o_out_vec = _fmul(o_norm_vec, sv_vec)
                o_trunc = Vec(o_out_vec).to(elem_out).ir_value()
                d_col = fx.Index(dc * D_CHUNK) + klane * 8
                o_global = out_global_idx(q_head, q_row, d_col)
                _store_global_out(o_ptr, o_global, o_trunc)





    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def _kernel_legacy(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,   # real S: kv loop bound, q in-bounds, store bound
        seq_pad: fx.Int32,   # padded S (=tensor layout stride); pad rows never read
        SCL: fx.Pointer,  # device [2] f32: qk_scale_log2e, scale_v (zero host sync)
    ):
        elem_type = T.i8  # LDS & input pointer element (fp8 bytes) — needs MLIR ctx
        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        o_ptr = _pointer_to_llvm_ptr(O)
        scl_ptr = _pointer_to_llvm_ptr(SCL)
        qk_scale_log2e = fx.Float32(
            _pointer_load(ir.F32Type.get(),
                          buffer_ops.get_element_ptr(scl_ptr, fx.Int64(0), elem_type=T.f32))
        )
        scale_v = fx.Float32(
            _pointer_load(ir.F32Type.get(),
                          buffer_ops.get_element_ptr(scl_ptr, fx.Int64(1), elem_type=T.f32))
        )
        fm_fast = arith.FastMathFlags.fast

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        v8f32_type = Vec.make_type(8, fx.Float32)
        v8i8_type = Vec.make_type(8, fx.Int8)
        vxf8_type = Vec.make_type(VEC_WIDTH, fx.Int8)
        v2f32_type = Vec.make_type(2, fx.Float32)

        def to_v2i32(x):
            return Vec(x).bitcast(fx.Int32)

        def wmma_fp8(a_v2i32, b_v2i32, c_v8f32):
            return rocdl.wmma_f32_16x16x16_fp8_fp8(
                v8f32_type, _raw(a_v2i32), _raw(b_v2i32), c_v8f32
            ).result

        seq_len_v = fx.Index(seq_len)

        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        wave_q_offset = wave_id * ROWS_PER_WAVE

        s_pad_v = fx.Index(seq_pad)
        kv_head_idx = block_id % NUM_KV_HEADS
        batch_q_tile_id = block_id // NUM_KV_HEADS
        num_q_tiles = (seq_len_v + ROWS_PER_HEAD - 1) // ROWS_PER_HEAD
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * ROWS_PER_HEAD

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def kv_global_idx(token_idx, col):
            # K/V: [B, NKV, S, D] BHSD contiguous (fp8 bytes)
            return ((batch_idx * NUM_KV_HEADS + kv_head_idx) * s_pad_v + token_idx) * HEAD_DIM + col

        def q_global_idx(q_head, q_s, col):
            # Q: [B, NQ, S, D] BHSD contiguous (fp8 bytes)
            return ((batch_idx * NUM_HEADS + q_head) * s_pad_v + q_s) * HEAD_DIM + col

        def out_global_idx(q_head, q_s, col):
            # O: [B, S, NQ*D] bf16, token-major (matches torchinfra's layout)
            return (batch_idx * s_pad_v + q_s) * STRIDE_TOKEN + q_head * HEAD_DIM + col

        def _load_global_vec(ptr, base_idx, vec_type):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(vec_type, gep)

        def _store_global_out(ptr, base_idx, val):
            # O is bf16: element indices are the same numbers but the GEP
            # element type (and thus the byte stride) is bf16
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=T.bf16)
            _pointer_store(val, gep)

        def _pack_fp8_pair(f_lo, f_hi, old_dword, word_sel):
            return rocdl.cvt_pk_fp8_f32(
                T.i32, _raw(fx.Float32(f_lo)), _raw(fx.Float32(f_hi)), _raw(old_dword), word_sel
            )

        def fp8_pack_v8_(f32_vals):
            c_zero_i32 = fx.Int32(0)
            d0 = _pack_fp8_pair(f32_vals[0], f32_vals[1], c_zero_i32, 0)
            d0 = _pack_fp8_pair(f32_vals[2], f32_vals[3], d0, 1)
            d1 = _pack_fp8_pair(f32_vals[4], f32_vals[5], c_zero_i32, 0)
            d1 = _pack_fp8_pair(f32_vals[6], f32_vals[7], d1, 1)
            return Vec.from_elements([d0, d1], fx.Int32).ir_value()

        def fp8_unpack_sum(dword):
            """Sum the 4 fp8 values inside one i32 dword (faithful l)."""
            va = rocdl.cvt_pk_f32_fp8(v2f32_type, _raw(dword), 0)
            vb = rocdl.cvt_pk_f32_fp8(v2f32_type, _raw(dword), 1)
            s = _fadd(Vec(va)[0], Vec(va)[1])
            s = _fadd(s, Vec(vb)[0])
            s = _fadd(s, Vec(vb)[1])
            return s

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)
            return fx.Index(LDS_V_BASE) + buf_id * fx.Index(LDS_V_TILE_SIZE)

        def coop_load_k_global(tile_start):
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = kv_global_idx(row_idx, load_col_base)
                vecs.append(_load_global_vec(k_ptr, g_idx, vxf8_type))
            return vecs

        def coop_store_k_lds(vecs, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                lds_idx = k_base + (load_row_in_batch + row_offset) * K_STRIDE + load_col_base
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        Vec(vecs[batch]).store(lds_kv, [lds_idx])
                else:
                    Vec(vecs[batch]).store(lds_kv, [lds_idx])

        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        g_idx = kv_global_idx(row_idx, load_col_base)
                        lds_idx = k_base + (load_row_in_batch + row_offset) * K_STRIDE + load_col_base
                        vec = _load_global_vec(k_ptr, g_idx, vxf8_type)
                        Vec(vec).store(lds_kv, [lds_idx])
                else:
                    g_idx = kv_global_idx(row_idx, load_col_base)
                    lds_idx = k_base + (load_row_in_batch + row_offset) * K_STRIDE + load_col_base
                    vec = _load_global_vec(k_ptr, g_idx, vxf8_type)
                    Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            Vec(vec).store(lds_kv, [lds_idx])

        def _v_store_transposed(v_base, lds_row, vec):
            # scatter VEC_WIDTH bytes of one KV row into [D][BN] layout:
            # 16 scalar 1B stores, but GEMM2 then reads 8B vectors
            for j in range_constexpr(VEC_WIDTH):
                d_col = load_col_base + j
                byte_idx = v_base + d_col * V_STRIDE + lds_row
                Vec.from_elements([Vec(vec)[j]], fx.Int8).store(lds_kv, [byte_idx])

        def coop_load_v_global(tile_start):
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = kv_global_idx(row_idx, load_col_base)
                vecs.append(_load_global_vec(v_ptr, g_idx, vxf8_type))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                store_v = _v_store_transposed if v_layout == "transposed" else _v_store_row_major
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
                    if row_valid:
                        store_v(v_base, load_row_in_batch + row_offset, vecs[batch])
                else:
                    store_v(v_base, load_row_in_batch + row_offset, vecs[batch])

        # ---- Q preload (8 fp8 bytes per kstep per lane) ----
        # GQA: this wave's 16 rows live inside ONE query head (16 | ROWS_PER_HEAD)
        # GQA row mapping: block row r -> (head_local = r//ROWS_PER_HEAD,
        # head-local seq = q_start + r%ROWS_PER_HEAD); wave never straddles
        # heads, so the div/mod are wave-uniform
        q_row = q_start + (wave_q_offset % fx.Index(ROWS_PER_HEAD)) + lane16
        q_head_local = wave_q_offset // fx.Index(ROWS_PER_HEAD)  # 0..NQ_PER_KV-1
        q_head = fx.Index(kv_head_idx * NQ_PER_KV) + q_head_local
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        c_zero_v8i8 = Vec.filled(8, 0, fx.Int8).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = q_global_idx(q_head, q_row_safe, q_col)
            raw = _load_global_vec(q_ptr, g_idx, v8i8_type)
            q_b_packs.append(ArithValue(q_in_bounds).select(raw, c_zero_v8i8))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_log2_pmax = fx.Float32(host_math.log2(448.0))  # fold into exp2 bias
        c_pmax = fx.Float32(448.0)  # e4m3 max; p<=1 so *448 never clips
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        # synth_p 诊断：循环外预打包的常数 P 片段（448.0 的 e4m3 编码），
        # 行数按 kv_merge 后的合并子 tile 数
        synth_p_packs = []
        if const_expr(ablate in ("synth_p", "neither")):
            for _st in range_constexpr(N_SUB_TILES * KV_MERGE):
                _row = []
                for _pks in range_constexpr(PV_K_STEPS):
                    _row.append(fp8_pack_v8_([c_pmax] * 8))
                synth_p_packs.append(_row)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)

        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_16_i32, width_i32)

        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v))
        else:
            kv_upper = seq_len_v
        _v_vecs_init = []
        for _t in range_constexpr(KV_MERGE):
            _v_vecs_init.append(coop_load_v_global(fx.Index(_t * BLOCK_N)))

        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(_raw(c_zero_v8f32))
        for _t in range_constexpr(KV_MERGE):
            for batch in range_constexpr(NUM_BATCHES_KV):
                init_args.append(_v_vecs_init[_t][batch])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            0, kv_upper, BLOCK_N_OUT * KV_MERGE, init=init_args
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _v_vecs_prefetch = [
                inner_iter_args[2 + D_CHUNKS + _t * NUM_BATCHES_KV + b]
                for _t in range_constexpr(KV_MERGE)
                for b in range_constexpr(NUM_BATCHES_KV)
            ]

            # ---- K 装载：KV_MERGE 个 tile 全部发完再一个 barrier ----
            for _t in range_constexpr(KV_MERGE):
                coop_load_k(kv_block_start + fx.Index(_t * BLOCK_N), _t)
            gpu.barrier()

            # ==== GEMM1: S = K @ Q^T (fp8 x fp8)，每 tile 独立可重叠 ====
            def _gemm1(k_base):
                accs = [_raw(c_zero_v8f32) for _ in range(NUM_S_ACCS)]
                for ks in range_constexpr(K_STEPS_QK):
                    k_col = fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K

                    for st_idx in range_constexpr(N_SUB_TILES):
                        st_base_row = st_idx * K_SUB_N

                        k_row_a = lane16 + fx.Index(st_base_row)
                        k_lds_a = k_base + k_row_a * K_STRIDE + k_col
                        k_pack_a = Vec.load(v8i8_type, lds_kv, [k_lds_a])

                        k_row_b = lane16 + fx.Index(st_base_row + 16)
                        k_lds_b = k_base + k_row_b * K_STRIDE + k_col
                        k_pack_b = Vec.load(v8i8_type, lds_kv, [k_lds_b])

                        accs[st_idx * 2] = wmma_fp8(
                            to_v2i32(k_pack_a), to_v2i32(q_b_packs[ks]), accs[st_idx * 2]
                        )
                        accs[st_idx * 2 + 1] = wmma_fp8(
                            to_v2i32(k_pack_b), to_v2i32(q_b_packs[ks]), accs[st_idx * 2 + 1]
                        )
                return accs

            s_accs_all = []
            for _t in range_constexpr(KV_MERGE):
                s_accs_all.append(_gemm1(k_buf_base(_t)))

            # ==== Online softmax (fp32，跨 KV_MERGE tile 合并) ====
            s_raw = []
            for _t in range_constexpr(KV_MERGE):
                for st in range_constexpr(NUM_S_ACCS):
                    for r in range_constexpr(8):
                        s_raw.append(Vec(s_accs_all[_t][st])[r])

            if const_expr(CAUSAL and KV_MERGE == 1):
                kv_start_i32 = fx.Int32(kv_block_start)
                klane_i32 = fx.Int32(klane)
                q_start_i32 = fx.Int32(q_start)
                max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                tile_needs_mask = max_kv_col_i32 > q_start_i32
                s_v0 = s_raw[0]
                s_v1 = s_raw[1]
                s_v2 = s_raw[2]
                s_v3 = s_raw[3]
                s_v4 = s_raw[4]
                s_v5 = s_raw[5]
                s_v6 = s_raw[6]
                s_v7 = s_raw[7]
                s_v8 = s_raw[8]
                s_v9 = s_raw[9]
                s_v10 = s_raw[10]
                s_v11 = s_raw[11]
                s_v12 = s_raw[12]
                s_v13 = s_raw[13]
                s_v14 = s_raw[14]
                s_v15 = s_raw[15]
                if tile_needs_mask:
                    klane_off_i32 = klane_i32 * fx.Int32(8)
                    _b0 = kv_start_i32 + fx.Int32(0) + klane_off_i32
                    s_v0 = ArithValue(_b0 > q_row_i32).select(c_neg_inf, s_v0)
                    _b1 = kv_start_i32 + fx.Int32(1) + klane_off_i32
                    s_v1 = ArithValue(_b1 > q_row_i32).select(c_neg_inf, s_v1)
                    _b2 = kv_start_i32 + fx.Int32(2) + klane_off_i32
                    s_v2 = ArithValue(_b2 > q_row_i32).select(c_neg_inf, s_v2)
                    _b3 = kv_start_i32 + fx.Int32(3) + klane_off_i32
                    s_v3 = ArithValue(_b3 > q_row_i32).select(c_neg_inf, s_v3)
                    _b4 = kv_start_i32 + fx.Int32(4) + klane_off_i32
                    s_v4 = ArithValue(_b4 > q_row_i32).select(c_neg_inf, s_v4)
                    _b5 = kv_start_i32 + fx.Int32(5) + klane_off_i32
                    s_v5 = ArithValue(_b5 > q_row_i32).select(c_neg_inf, s_v5)
                    _b6 = kv_start_i32 + fx.Int32(6) + klane_off_i32
                    s_v6 = ArithValue(_b6 > q_row_i32).select(c_neg_inf, s_v6)
                    _b7 = kv_start_i32 + fx.Int32(7) + klane_off_i32
                    s_v7 = ArithValue(_b7 > q_row_i32).select(c_neg_inf, s_v7)
                    _b8 = kv_start_i32 + fx.Int32(16) + klane_off_i32
                    s_v8 = ArithValue(_b8 > q_row_i32).select(c_neg_inf, s_v8)
                    _b9 = kv_start_i32 + fx.Int32(17) + klane_off_i32
                    s_v9 = ArithValue(_b9 > q_row_i32).select(c_neg_inf, s_v9)
                    _b10 = kv_start_i32 + fx.Int32(18) + klane_off_i32
                    s_v10 = ArithValue(_b10 > q_row_i32).select(c_neg_inf, s_v10)
                    _b11 = kv_start_i32 + fx.Int32(19) + klane_off_i32
                    s_v11 = ArithValue(_b11 > q_row_i32).select(c_neg_inf, s_v11)
                    _b12 = kv_start_i32 + fx.Int32(20) + klane_off_i32
                    s_v12 = ArithValue(_b12 > q_row_i32).select(c_neg_inf, s_v12)
                    _b13 = kv_start_i32 + fx.Int32(21) + klane_off_i32
                    s_v13 = ArithValue(_b13 > q_row_i32).select(c_neg_inf, s_v13)
                    _b14 = kv_start_i32 + fx.Int32(22) + klane_off_i32
                    s_v14 = ArithValue(_b14 > q_row_i32).select(c_neg_inf, s_v14)
                    _b15 = kv_start_i32 + fx.Int32(23) + klane_off_i32
                    s_v15 = ArithValue(_b15 > q_row_i32).select(c_neg_inf, s_v15)
                s_raw = [
                    s_v0, s_v1, s_v2, s_v3, s_v4, s_v5, s_v6, s_v7,
                    s_v8, s_v9, s_v10, s_v11, s_v12, s_v13, s_v14, s_v15,
                ]

            local_max = s_raw[0]
            for r in range_constexpr(NUM_S_VALS_M - 1):
                local_max = _fmax(local_max, s_raw[r + 1])
            peer_max = reduction_peer(local_max)
            row_max = _fmax(local_max, peer_max)
            m_new_raw = _fmax(m_running, row_max)

            diff_m_raw = _fsub(m_running, m_new_raw)
            diff_m_scaled = _fmul(diff_m_raw, qk_scale_log2e)
            corr = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_scaled))

            l_corr = _fmul(corr, l_running)

            if const_expr(ablate in ("synth_p", "neither")):
                # 诊断：不构造 P（省 exp2/fma/sum 链），l 链只留 corr 缩放
                l_new = l_corr
            else:
                scaled_max = _fmul(qk_scale_log2e, m_new_raw)
                neg_scaled_max = _fsub(c_zero_f, scaled_max)

                p_vals = []
                local_sum = _raw(c_zero_f)
                for r in range_constexpr(NUM_S_VALS_M):
                    diff = fmath.fma(_raw(s_raw[r]), _raw(qk_scale_log2e), neg_scaled_max)
                    p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                    p_vals.append(p)
                    local_sum = _fadd(local_sum, p)

                peer_sum = reduction_peer(local_sum)
                tile_sum = _fadd(local_sum, peer_sum)
                l_new = _fadd(l_corr, tile_sum)

            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

            for _t in range_constexpr(KV_MERGE):
                coop_store_v_lds(
                    _v_vecs_prefetch[_t * NUM_BATCHES_KV:(_t + 1) * NUM_BATCHES_KV], _t
                )
            gpu.barrier()

            # ==== Build P packs: p*448 -> e4m3 (v2i32) ====
            if const_expr(ablate in ("synth_p", "neither")):
                p_packs_all = synth_p_packs
            else:
                p_packs_all = []
                p_dwords = []  # keep dwords for the faithful denominator
                for _t in range_constexpr(KV_MERGE):
                    for st_idx in range_constexpr(N_SUB_TILES):
                        p_packs_st = []
                        for pks in range_constexpr(PV_K_STEPS):
                            acc_idx = _t * NUM_S_ACCS + st_idx * 2 + pks
                            p_base = acc_idx * 8
                            p_slice = [p_vals[p_base + j] for j in range(8)]
                            scaled = [_fmul(p_slice[j], c_pmax) for j in range(8)]
                            pack = fp8_pack_v8_(scaled)
                            p_packs_st.append(pack)
                            p_dwords.append(pack)
                        p_packs_all.append(p_packs_st)

                if const_expr(FAITHFUL_L):
                    # faithful l: sum the dequantized quantized-P (same domain
                    # as the bytes actually fed to PV — quantization-rounding-
                    # -consistent)
                    local_lq = _raw(c_zero_f)
                    for d_idx in range_constexpr(N_SUB_TILES * KV_MERGE * PV_K_STEPS):
                        local_lq = _fadd(local_lq, fp8_unpack_sum(Vec(p_dwords[d_idx])[0]))
                        local_lq = _fadd(local_lq, fp8_unpack_sum(Vec(p_dwords[d_idx])[1]))
                    peer_lq = reduction_peer(local_lq)
                    tile_lq = _fadd(local_lq, peer_lq)
                    l_new = _fadd(l_corr, tile_lq)
                else:
                    # cheap l: 直接对 exp2 域求和再补 ×448（分子在量化域，
                    # 分母必须同域；省 8 cvt 解码 + 16 加/tile）
                    l_new = _fadd(l_corr, _fmul(tile_sum, c_pmax))

            # ==== GEMM2: O += V^T @ P (fp8 x fp8, software pipelined) ====
            def _load_v_rowmajor(v_base, st_kv_base_val, pks_val, dc_val):
                d_pos = fx.Index(dc_val * D_CHUNK) + lane16
                v_elems = []
                for k_sub in range_constexpr(8):
                    kv_row = (
                        fx.Index(st_kv_base_val + pks_val * PV_K_STEP)
                        + klane * WMMA_LANE_K
                        + fx.Index(k_sub)
                    )
                    v_lds_idx = v_base + kv_row * V_STRIDE + d_pos
                    v_elems.append(_memref.load(lds_kv, [_raw(v_lds_idx)]))
                return Vec.from_elements(v_elems, fx.Int8).ir_value()

            def _load_v_transposed(v_base, st_kv_base_val, pks_val, dc_val):
                # [D][BN] tile: 8 consecutive K positions of one d = 8B vector
                d_pos = fx.Index(dc_val * D_CHUNK) + lane16
                k0 = fx.Index(st_kv_base_val + pks_val * PV_K_STEP) + klane * WMMA_LANE_K
                v_lds_idx = v_base + d_pos * V_STRIDE + k0
                return Vec.load(v8i8_type, lds_kv, [v_lds_idx])

            load_v = _load_v_transposed if v_layout == "transposed" else _load_v_rowmajor

            for _t in range_constexpr(KV_MERGE):
                v_base = v_buf_base(_t)
                cur_v_packs = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    cur_v_packs.append(load_v(v_base, st_idx * K_SUB_N, 0, 0))

                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        next_dc = dc + 1
                        next_pks = pks
                        if const_expr(next_dc >= D_CHUNKS):
                            next_dc = 0
                            next_pks = pks + 1
                        has_next = const_expr(next_pks < PV_K_STEPS)

                        next_v_packs = []
                        if const_expr(has_next):
                            for st_idx in range_constexpr(N_SUB_TILES):
                                next_v_packs.append(
                                    load_v(v_base, st_idx * K_SUB_N, next_pks, next_dc)
                                )

                        for st_idx in range_constexpr(N_SUB_TILES):
                            if const_expr(ablate not in ("no_pv", "neither")):
                                o_accs[dc] = wmma_fp8(
                                    to_v2i32(cur_v_packs[st_idx]),
                                    to_v2i32(p_packs_all[_t * N_SUB_TILES + st_idx][pks]),
                                    o_accs[dc],
                                )

                        if const_expr(has_next):
                            cur_v_packs = next_v_packs

            m_running = m_new_raw
            l_running = l_new

            next_kv_start = kv_block_start + fx.Index(BLOCK_N_OUT * KV_MERGE)
            _yield_args = [m_running, l_running] + o_accs
            for _t in range_constexpr(KV_MERGE):
                _tile_vecs = coop_load_v_global(next_kv_start + fx.Index(_t * BLOCK_N))
                for batch in range_constexpr(NUM_BATCHES_KV):
                    _yield_args.append(_tile_vecs[batch])
            loop_results = yield _yield_args
        # ---- Normalize and store O (bf16 out, descale by scale_v) ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()
        sv_vec = Vec.from_elements([scale_v], fx.Float32).broadcast_to(8).ir_value()

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = _fmul(o_finals[dc], inv_l_vec)
                o_out_vec = _fmul(o_norm_vec, sv_vec)
                o_trunc = Vec(o_out_vec).to(elem_out).ir_value()
                d_col = fx.Index(dc * D_CHUNK) + klane * 8
                o_global = out_global_idx(q_head, q_row, d_col)
                _store_global_out(o_ptr, o_global, o_trunc)




    if SW_PIPE:
        @flyc.jit
        def launch_flash_attn_fp8(
            Q: fx.Pointer,
            K: fx.Pointer,
            V: fx.Pointer,
            O: fx.Pointer,  # noqa: E741
            batch_size: fx.Int32,
            seq_len: fx.Int32,
            seq_pad: fx.Int32,
            SCL: fx.Pointer,
            stream: fx.Stream = fx.Stream(None),
        ):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()

            bs_idx = fx.Index(batch_size)
            sl_idx = fx.Index(seq_len)
            sp_idx = fx.Index(seq_pad)
            num_q_tiles = (sl_idx + ROWS_PER_HEAD - 1) // ROWS_PER_HEAD
            grid_x = bs_idx * num_q_tiles * NUM_KV_HEADS

            launcher = _kernel_pipe(Q, K, V, O, seq_len, seq_pad, SCL)

            if const_expr(waves_per_eu is not None):
                _wpe = int(waves_per_eu)
                if const_expr(_wpe >= 1):
                    for op in ctx.gpu_module_body.operations:
                        if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                            op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
            if const_expr(flat_work_group_size is not None):
                _fwgs = int(flat_work_group_size)
                if const_expr(_fwgs >= 1):
                    flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                    for op in ctx.gpu_module_body.operations:
                        if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                            op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

            passthrough_entries = []
            if const_expr(daz):
                passthrough_entries.append(
                    ir.ArrayAttr.get(
                        [
                            ir.StringAttr.get("denormal-fp-math-f32"),
                            ir.StringAttr.get("preserve-sign,preserve-sign"),
                        ]
                    )
                )
                passthrough_entries.append(
                    ir.ArrayAttr.get([ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")])
                )
                passthrough_entries.append(
                    ir.ArrayAttr.get([ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")])
                )
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

            launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

        _fmha_compile_hints = {
            "fast_fp_math": fast_fp_math,
            "unsafe_fp_math": unsafe_fp_math,
            "llvm_options": {"enable-post-misched": False},
        }

        def _ptr_arg(t):
            if hasattr(t, "data_ptr"):
                type_name = type(t).__name__
                module_name = type(t).__module__
                ptr = 0 if type_name == "FakeTensor" or "fake_tensor" in module_name else t.data_ptr()
                return flyc.from_c_void_p(fx.Uint8, ptr)
            return t

        def _wrap_qkvo(args, kwargs):
            # 0-3 = Q,K,V,O; 7 = SCL (tensors→raw pointers); 4/5/6 are ints
            args = list(args)
            for idx in list(range(4)) + [7]:
                args[idx] = _ptr_arg(args[idx])
            for name in ("Q", "K", "V", "O"):
                if name in kwargs:
                    kwargs[name] = _ptr_arg(kwargs[name])
            return tuple(args), kwargs

        launch_flash_attn_fp8.compile_hints = dict(_fmha_compile_hints)

        def _launch(*args, **kwargs):
            args, kwargs = _wrap_qkvo(args, kwargs)
            stream = kwargs.pop("stream", fx.Stream(None))
            _run_compiled(launch_flash_attn_fp8, *args, stream=stream)

        def _compile(Q, K, V, O, batch_size, seq_len, seq_pad, scl, stream=None):  # noqa: E741
            return flyc.compile(
                launch_flash_attn_fp8,
                _ptr_arg(Q),
                _ptr_arg(K),
                _ptr_arg(V),
                _ptr_arg(O),
                batch_size,
                seq_len,
                seq_pad,
                _ptr_arg(scl),
                fx.Stream(stream),
            )

        _launch.compile = _compile
        return _launch

    else:
        @flyc.jit
        def launch_flash_attn_fp8(
            Q: fx.Pointer,
            K: fx.Pointer,
            V: fx.Pointer,
            O: fx.Pointer,  # noqa: E741
            batch_size: fx.Int32,
            seq_len: fx.Int32,
            seq_pad: fx.Int32,
            SCL: fx.Pointer,
            stream: fx.Stream = fx.Stream(None),
        ):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()

            bs_idx = fx.Index(batch_size)
            sl_idx = fx.Index(seq_len)
            sp_idx = fx.Index(seq_pad)
            num_q_tiles = (sl_idx + ROWS_PER_HEAD - 1) // ROWS_PER_HEAD
            grid_x = bs_idx * num_q_tiles * NUM_KV_HEADS

            launcher = _kernel_legacy(Q, K, V, O, seq_len, seq_pad, SCL)

            if const_expr(waves_per_eu is not None):
                _wpe = int(waves_per_eu)
                if const_expr(_wpe >= 1):
                    for op in ctx.gpu_module_body.operations:
                        if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                            op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
            if const_expr(flat_work_group_size is not None):
                _fwgs = int(flat_work_group_size)
                if const_expr(_fwgs >= 1):
                    flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                    for op in ctx.gpu_module_body.operations:
                        if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                            op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

            passthrough_entries = []
            if const_expr(daz):
                passthrough_entries.append(
                    ir.ArrayAttr.get(
                        [
                            ir.StringAttr.get("denormal-fp-math-f32"),
                            ir.StringAttr.get("preserve-sign,preserve-sign"),
                        ]
                    )
                )
                passthrough_entries.append(
                    ir.ArrayAttr.get([ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")])
                )
                passthrough_entries.append(
                    ir.ArrayAttr.get([ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")])
                )
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

            launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

        _fmha_compile_hints = {
            "fast_fp_math": fast_fp_math,
            "unsafe_fp_math": unsafe_fp_math,
            "llvm_options": {"enable-post-misched": False},
        }

        def _ptr_arg(t):
            if hasattr(t, "data_ptr"):
                type_name = type(t).__name__
                module_name = type(t).__module__
                ptr = 0 if type_name == "FakeTensor" or "fake_tensor" in module_name else t.data_ptr()
                return flyc.from_c_void_p(fx.Uint8, ptr)
            return t

        def _wrap_qkvo(args, kwargs):
            # 0-3 = Q,K,V,O; 7 = SCL (tensors→raw pointers); 4/5/6 are ints
            args = list(args)
            for idx in list(range(4)) + [7]:
                args[idx] = _ptr_arg(args[idx])
            for name in ("Q", "K", "V", "O"):
                if name in kwargs:
                    kwargs[name] = _ptr_arg(kwargs[name])
            return tuple(args), kwargs

        launch_flash_attn_fp8.compile_hints = dict(_fmha_compile_hints)

        def _launch(*args, **kwargs):
            args, kwargs = _wrap_qkvo(args, kwargs)
            stream = kwargs.pop("stream", fx.Stream(None))
            _run_compiled(launch_flash_attn_fp8, *args, stream=stream)

        def _compile(Q, K, V, O, batch_size, seq_len, seq_pad, scl, stream=None):  # noqa: E741
            return flyc.compile(
                launch_flash_attn_fp8,
                _ptr_arg(Q),
                _ptr_arg(K),
                _ptr_arg(V),
                _ptr_arg(O),
                batch_size,
                seq_len,
                seq_pad,
                _ptr_arg(scl),
                fx.Stream(stream),
            )

        _launch.compile = _compile
        return _launch
