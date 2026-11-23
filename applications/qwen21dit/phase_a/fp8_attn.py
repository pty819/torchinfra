# -*- coding: utf-8 -*-
"""FlyDSL fp8 flash attention 接入 (torchinfra flydsl_kernels 移植, gfx1201)
- tensorwise 量化 q/k/v → 双 wmma_f32_16x16x16_fp8_fp8 → bf16 输出 [B,S,H*D]
- torch.library.custom_op 包装: dynamo 可追踪, compile 不图断裂
- 开关: 环境变量 QDIT_FP8_ATTN=1"""
from paths import TORCHINFRA_DIR
import os, math, torch
import torch.nn.functional as F

ENABLED = os.environ.get("QDIT_FP8_ATTN", "0") == "1"
_EXE = None

def _get_exe(H=32, D=128):
    global _EXE
    if _EXE is None:
        import sys
        sys.path.insert(0, str(TORCHINFRA_DIR))
        from flydsl_kernels.flash_attn_fp8_gfx1201 import build_flash_attn_fp8_module
        _EXE = build_flash_attn_fp8_module(num_heads=H, head_dim=D, num_kv_heads=H,
                                           causal=False, waves_per_eu=1,
                                           block_m=None, block_n=None,
                                           v_layout="transposed")
    return _EXE

def _quant(t):
    """tensorwise e4m3: 返回 (q8, scale)"""
    amax = t.float().abs().amax().clamp(min=1e-12)
    scale = amax / 448.0
    q = (t.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale

@torch.library.custom_op("qdit::fp8_flash_attn", mutates_args=())
def fp8_flash_attn(q8: torch.Tensor, k8: torch.Tensor, v8: torch.Tensor,
                   scl: torch.Tensor, S_real: int) -> torch.Tensor:
    """q8/k8/v8: [B,H,S_pad,D] e4m3, S_pad 已是 128 倍数; scl: [2] f32 (qk_scale_log2e, scale_v)
    返回 [B,S_pad,H*D] bf16 (调用方切片到 S_real; 真实 S 传内核, pad 不进 softmax 分母)"""
    B, H, S_pad, D = q8.shape
    assert S_pad % 128 == 0
    o = torch.empty(B, S_pad, H * D, dtype=torch.bfloat16, device=q8.device)
    _get_exe(H, D)(q8.view(-1).view(torch.int8), k8.view(-1).view(torch.int8),
                   v8.view(-1).view(torch.int8), o.view(-1), B, S_real, S_pad, scl,
                   stream=torch.cuda.current_stream(q8.device))
    return o

@fp8_flash_attn.register_fake
def _(q8, k8, v8, scl, S_real):
    B, H, S_pad, D = q8.shape
    return torch.empty(B, S_pad, H * D, dtype=torch.bfloat16, device=q8.device)

def _pad8(t8, S_pad):
    pad = S_pad - t8.shape[2]
    if pad == 0:
        return t8
    return F.pad(t8.view(torch.int8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)

def attention_fp8(q, k, v):
    """q/k/v: [B,S,H,D] bf16 → fp8 attention → [B,S,H*D] bf16
    pad/量化都留在 custom_op 外: inductor 可融进上游 epilogue"""
    B, S, H, D = q.shape
    S_pad = (S + 127) // 128 * 128
    q8, sq = _quant(q.permute(0, 2, 1, 3))
    k8, sk = _quant(k.permute(0, 2, 1, 3))
    v8, sv = _quant(v.permute(0, 2, 1, 3))
    qk_log2e = (sq / math.sqrt(D)) * sk * math.log2(math.e)
    scl = torch.stack([qk_log2e, sv]).float()
    o = fp8_flash_attn(_pad8(q8, S_pad), _pad8(k8, S_pad), _pad8(v8, S_pad), scl, S)
    return o[:, :S]
