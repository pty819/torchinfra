# wsplit2 S 扫描二分：找出误差开始复合的位置
import sys
import torch, torch.nn.functional as F
from flydsl_kernels.fp8_wrapper import quantize_tensorwise
from flydsl_kernels.flash_attn_fp8_wsplit2 import build_flash_attn_fp8_wsplit2_module

D = 128
exe = build_flash_attn_fp8_wsplit2_module(num_heads=32, head_dim=D, num_kv_heads=8)

def run(S, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    B, NQ, NKV = 1, 32, 8
    q = (torch.randn(B, NQ, S, D, device='cuda', generator=g) * 0.3).bfloat16()
    k = (torch.randn(B, NKV, S, D, device='cuda', generator=g) * 0.3).bfloat16()
    v = (torch.randn(B, NKV, S, D, device='cuda', generator=g) * 0.3).bfloat16()
    q8, sq = quantize_tensorwise(q); k8, sk = quantize_tensorwise(k); v8, sv = quantize_tensorwise(v)
    ref = F.scaled_dot_product_attention(q8.float() * sq, k8.float() * sk, v8.float() * sv,
                                         enable_gqa=True)
    S_pad = (S + 127) // 128 * 128
    pad = S_pad - S
    ic8 = torch.int8
    if pad:
        q_p = F.pad(q8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        k_p = F.pad(k8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        v_p = F.pad(v8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        for t in (k_p, v_p):
            junk = torch.randint(-128, 128, tuple(t[:, :, S:, :].shape), device='cuda', dtype=torch.int8)
            junk.view(-1)[::97] = 127
            t.view(torch.int8)[:, :, S:, :] = junk
    else:
        q_p, k_p, v_p = q8, k8, v8
    o_p = torch.empty(B, S_pad, NQ * D, dtype=torch.bfloat16, device='cuda')
    import math
    scl = torch.tensor([sq * sk / math.sqrt(D) * math.log2(math.e), sv],
                       dtype=torch.float32, device='cuda')
    exe(q_p.view(-1).view(ic8), k_p.view(-1).view(ic8), v_p.view(-1).view(ic8),
        o_p.view(-1), B, S, S_pad, scl)
    torch.cuda.synchronize()
    d = (o_p[:, :S].view(1, S, NQ, D).permute(0, 2, 1, 3).float() - ref).abs()
    return d.max().item(), d.mean().item()

for S in [64, 128, 256, 384, 512, 1024, 2048]:
    mx, mn = run(S)
    print(f"S={S:5d}: Δmax={mx:.5f} Δmean={mn:.6f}")
print("DONE")
