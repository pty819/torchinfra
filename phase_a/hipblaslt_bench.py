#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裸调 hipBLASLt fp8 rowwise GEMM 对照基准 (ctypes, 绕过 torch dispatch)
目的: 1) 验证 torch._scaled_mm 的 ~99TF/s 是否就是 kernel 本体速度
      2) 扫启发式 algo 列表, 看 torch 默认选择之外还有没有余量
布局推导(行主序→列主序换轴): D_rm(M,N) = A_rm(M,K) × B_rm(N,K)^T
  ⇒ D'_cm(N,M) = opT(B_rm) × opN(A_rm), 尺度: A_SCALE←sb(N), B_SCALE←sa(M)"""
import ctypes as C
import torch, time, sys

LIB = "~/rocm10/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/libhipblaslt.so.1"
lt = C.CDLL(LIB)

# ---- 常量(从头文件抄) ----
R32F, R16BF, E4M3 = 0, 14, 28
COMPUTE_32F = 2                        # hipblas-common.h (非 cuBLAS 的 68)
OP_N, OP_T = 0, 1
DESC_TRANSA, DESC_TRANSB = 0, 1
DESC_A_SCALE_PTR, DESC_B_SCALE_PTR = 5, 6
DESC_A_SCALE_MODE, DESC_B_SCALE_MODE = 31, 32
SCALE_OUTER_VEC_32F = 3
PREF_MAX_WORKSPACE = 1

class Algo(C.Structure):
    _fields_ = [("data", C.c_uint8 * 16), ("max_workspace_bytes", C.c_size_t)]

class Heur(C.Structure):
    _fields_ = [("algo", Algo), ("workspaceSize", C.c_size_t), ("state", C.c_int),
                ("wavesCount", C.c_float), ("reserved", C.c_int * 4)]

P, VP = C.c_void_p, C.byref
def ck(st, tag):
    if st != 0:
        raise RuntimeError(f"{tag} -> hipblasStatus {st}")

h = VP(C.c_void_p())
ck(lt.hipblasLtCreate(h), "create"); h = h._obj

def desc_create():
    d = VP(C.c_void_p())
    ck(lt.hipblasLtMatmulDescCreate(d, COMPUTE_32F, R32F), "descCreate")
    return d._obj

def desc_set_i32(d, attr, v):
    ck(lt.hipblasLtMatmulDescSetAttribute(d, attr, VP(C.c_int(v)), 4), f"setAttr{attr}")

def layout(dt, rows, cols, ld):
    l = VP(C.c_void_p())
    ck(lt.hipblasLtMatrixLayoutCreate(l, dt, C.c_int32(rows), C.c_int32(cols), C.c_int64(ld)), "layout")
    return l._obj

def gemm_call(a, sa, b, sb, d, M, N, K, algo, ws, ws_size, stream):
    """D_rm(M,N) = A_rm(M,K)@B_rm(N,K)^T (fp8×fp8→bf16, rowwise 尺度)"""
    op = desc_create()
    desc_set_i32(op, DESC_TRANSA, OP_T)          # A 操作数 = B 缓冲 (K×N, ld=K)
    desc_set_i32(op, DESC_TRANSB, OP_N)          # B 操作数 = A 缓冲 (K×M, ld=K)
    la = layout(E4M3, K, N, K)                   # rows,cols,ld (列主序)
    lb = layout(E4M3, K, M, K)
    lc = layout(R16BF, N, M, N)
    ld_ = layout(R16BF, N, M, N)
    ck(lt.hipblasLtMatmulDescSetAttribute(op, DESC_A_SCALE_PTR, VP(C.c_void_p(sb.data_ptr())), 8), "aScalePtr")
    ck(lt.hipblasLtMatmulDescSetAttribute(op, DESC_B_SCALE_PTR, VP(C.c_void_p(sa.data_ptr())), 8), "bScalePtr")
    desc_set_i32(op, DESC_A_SCALE_MODE, SCALE_OUTER_VEC_32F)
    desc_set_i32(op, DESC_B_SCALE_MODE, SCALE_OUTER_VEC_32F)
    alpha, beta = C.c_float(1.0), C.c_float(0.0)
    st = lt.hipblasLtMatmul(h, op, C.byref(alpha),
                            C.c_void_p(b.data_ptr()), la,
                            C.c_void_p(a.data_ptr()), lb,
                            C.byref(beta),
                            C.c_void_p(d.data_ptr()), lc,
                            C.c_void_p(d.data_ptr()), ld_,
                            algo, C.c_void_p(ws.data_ptr() if ws is not None else 0),
                            C.c_size_t(ws_size or 0),
                            C.c_void_p(stream))
    ck(st, "matmul")

def run_shape(M, N, K, torch_cmp=True):
    torch.manual_seed(0)
    dev = "cuda"
    a8 = (torch.randn(M, K, device=dev) * 0.1).clamp(-0.3, 0.3).to(torch.float8_e4m3fn)
    b8 = (torch.randn(N, K, device=dev) * 0.1).clamp(-0.3, 0.3).to(torch.float8_e4m3fn)
    sa = torch.rand(M, 1, device=dev) * 0.05 + 0.01
    sb = torch.rand(1, N, device=dev) * 0.05 + 0.01
    ws = torch.empty(64 * 2**20, dtype=torch.uint8, device=dev)

    # ---- 启发式扫描 (algo 必须从这里拿, 零值 struct 无效) ----
    pref = VP(C.c_void_p())
    ck(lt.hipblasLtMatmulPreferenceCreate(pref), "prefCreate"); pref = pref._obj
    ck(lt.hipblasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE,
        VP(C.c_uint64(ws.numel())), 8), "prefWS")
    # 建一个 op desc 拿 heuristic
    opq = desc_create()
    desc_set_i32(opq, DESC_TRANSA, OP_T); desc_set_i32(opq, DESC_TRANSB, OP_N)
    la = layout(E4M3, K, N, K); lb = layout(E4M3, K, M, K)
    lc = layout(R16BF, N, M, N)
    heurs = (Heur * 24)()
    nret = C.c_int(0)
    ck(lt.hipblasLtMatmulAlgoGetHeuristic(h, opq, la, lb, lc, lc, pref, 24, heurs, VP(nret)), "heuristic")
    n = nret.value
    print(f"  启发式可用 algo: {n} 个")
    d = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    stream = torch.cuda.current_stream().cuda_stream

    # ---- 正确性(小形状才做): 用 heuristic[0] 的 algo vs f32 反量化参考 ----
    if M * N <= 2**22 and n > 0:
        ref = (a8.float() * sa) @ (b8.float() * sb.t()).t()
        gemm_call(a8, sa, b8, sb, d, M, N, K, heurs[0].algo, ws, ws.numel(), stream)
        torch.cuda.synchronize()
        err = (d.float() - ref).abs().max().item()
        scale = ref.abs().max().item()
        print(f"  正确性: max|Δ|={err:.5f} (参考峰值 {scale:.3f}, 相对 {err/max(scale,1e-6):.4f})")
        assert err / max(scale, 1e-6) < 0.05, "数值不对 → 布局反了!"
    fl = 2.0 * M * N * K

    def bench(fn, iters=30):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters

    results = []
    for i in range(min(n, 24)):
        if heurs[i].state != 0:
            continue
        algo = heurs[i].algo
        try:
            t = bench(lambda: gemm_call(a8, sa, b8, sb, d, M, N, K, algo, ws, ws.numel(), stream))
            results.append((t, i, heurs[i].wavesCount))
        except RuntimeError as e:
            print(f"    algo[{i}] 失败: {e}")
    results.sort()
    for t, i, w in results[:5]:
        print(f"    algo[{i:2d}]: {t*1000:7.3f}ms  {fl/t/1e12:6.1f} TF  waves={w:.2f}")
    best = results[0]

    if torch_cmp:
        sa2 = sa.reshape(M, 1).contiguous(); sb2 = sb.reshape(1, N).contiguous().t().contiguous()
        b8t = b8.t()
        t = bench(lambda: torch._scaled_mm(a8, b8t, scale_a=sa2, scale_b=sb2.t(), out_dtype=torch.bfloat16))
        print(f"    torch._scaled_mm: {t*1000:7.3f}ms  {fl/t/1e12:6.1f} TF")
        tt = t
    else:
        tt = None
    return best, tt

print("=== 小形状正确性 (M=256, N=256, K=512) ===")
run_shape(256, 256, 512)
print("=== 真实形状 1: M=7102, K=4096, N=12288 (MLP gate/up) ===")
run_shape(7102, 12288, 4096)
print("=== 真实形状 2: M=7102, K=4096, N=4096 (attn qkvo) ===")
run_shape(7102, 4096, 4096)
print("=== 真实形状 3: M=7102, K=12288, N=4096 (mlp down) ===")
run_shape(7102, 4096, 12288)
