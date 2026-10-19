"""decode 路径 CUDA Graph 化：GPU 侧 cache 写入 kernel + 整步图捕获（vLLM 式）。

三件事消灭 decode 的 CPU 开销：
1. append_kv kernel：cache 写入全在 GPU（索引来自 GPU 张量，无 int() 同步）
2. 静态缓冲 + pad：token/block_table/kv_lens 固定形状缓冲，pad 槽位指向预留的
   dummy 块 0（写入被丢弃，attention 读到零值、结果被丢弃）
3. torch.cuda.CUDAGraph 捕获整个 decode step：一次 replay 替代数百次 kernel launch
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _append_kv_kernel(K_cache, V_cache, K_new, V_new, block_table, kv_lens,
                      s_kc_blk, s_kc_h,
                      s_kn_b, s_kn_h, s_vn_b, s_vn_h,
                      s_bt_b, s_bt_s,
                      HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(kv_lens + b)                       # GPU 读取，无同步
    blk = tl.load(block_table + b * s_bt_b
                  + (pos // BLOCK) * s_bt_s).to(tl.int64)
    blk = tl.maximum(blk, 0)   # 未分配槽位是 -1：钳到 dummy 块 0，防狂野写入
    slot = pos % BLOCK
    offs_d = tl.arange(0, HEAD_DIM)
    kn = tl.load(K_new + b * s_kn_b + h * s_kn_h + offs_d)
    vn = tl.load(V_new + b * s_vn_b + h * s_vn_h + offs_d)
    tl.store(K_cache + blk * s_kc_blk + h * s_kc_h + slot * HEAD_DIM + offs_d,
             kn.to(K_cache.dtype.element_ty))
    tl.store(V_cache + blk * s_kc_blk + h * s_kc_h + slot * HEAD_DIM + offs_d,
             vn.to(V_cache.dtype.element_ty))


def append_kv(k_cache: torch.Tensor, v_cache: torch.Tensor,
              k_new: torch.Tensor, v_new: torch.Tensor,
              block_table: torch.Tensor, kv_lens: torch.Tensor,
              block_size: int, layer_idx: int = 0) -> None:
    """把 k_new/v_new [B, n_kv, 1, D] 写入各请求 kv_len 指示的槽位。

    k/v_new 允许 transposed 视图，只要 head_dim 维连续（按 stride 寻址）。
    不更新 kv_lens（由调用方统一管理，图内为 add_(1)）。
    """
    kc = k_cache[layer_idx]
    vc = v_cache[layer_idx]
    B, n_kv = k_new.shape[0], k_new.shape[1]
    _append_kv_kernel[(B, n_kv)](
        kc, vc, k_new, v_new, block_table, kv_lens,
        kc.stride(0), kc.stride(1),
        k_new.stride(0), k_new.stride(1),
        v_new.stride(0), v_new.stride(1),
        block_table.stride(0), block_table.stride(1),
        HEAD_DIM=k_new.shape[-1], BLOCK=block_size)
