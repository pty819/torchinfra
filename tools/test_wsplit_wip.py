import sys
import torch, torch.nn.functional as F
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl_kernels.fp8_wrapper import quantize_tensorwise

def flydsl_wsplit(q8, k8, v8, scale_q, scale_k, scale_v, num_kv_heads):
    import math, torch
    from flydsl_kernels.flash_attn_fp8_wsplit import build_flash_attn_fp8_wsplit_module
    B, NQ, S, D = q8.shape
    nkv = num_kv_heads
    S_pad = (S + 127) // 128 * 128
    pad = S_pad - S
    ic8 = torch.int8
    if pad:
        q_p = F.pad(q8.view(ic8), (0,0,0,pad,0,0)).view(torch.float8_e4m3fn)
        k_p = F.pad(k8.view(ic8), (0,0,0,pad,0,0)).view(torch.float8_e4m3fn)
        v_p = F.pad(v8.view(ic8), (0,0,0,pad,0,0)).view(torch.float8_e4m3fn)
    else:
        q_p, k_p, v_p = q8, k8, v8
    o_p = torch.empty(B, S_pad, NQ*D, dtype=torch.bfloat16, device=q8.device)
    scl = torch.tensor([scale_q*scale_k/math.sqrt(D)*math.log2(math.e), scale_v],
                       dtype=torch.float32, device=q8.device)
    exe = build_flash_attn_fp8_wsplit_module(num_heads=NQ, head_dim=D, num_kv_heads=nkv, dbg=bool(int(__import__('os').environ.get('WSDBG','0'))))
    exe(q_p.view(-1).view(ic8), k_p.view(-1).view(ic8), v_p.view(-1).view(ic8),
        o_p.view(-1), B, S, S_pad, scl)
    return o_p[:, :S].contiguous() if pad else o_p

torch.manual_seed(0)
B,NQ,NKV,D = 1,32,8,128
S = int(__import__('os').environ.get('WSS','512'))
q = (torch.randn(B,NQ,S,D,device='cuda')*0.3).bfloat16()
k = (torch.randn(B,NKV,S,D,device='cuda')*0.3).bfloat16()
v = (torch.randn(B,NKV,S,D,device='cuda')*0.3).bfloat16()
q8,sq = quantize_tensorwise(q); k8,sk = quantize_tensorwise(k); v8,sv = quantize_tensorwise(v)
ref = F.scaled_dot_product_attention(q8.float()*sq, k8.float()*sk, v8.float()*sv, enable_gqa=True)
o = flydsl_wsplit(q8,k8,v8,sq,sk,sv,NKV)
torch.cuda.synchronize()
d = (o.view(B,S,NQ,D).permute(0,2,1,3).float()-ref).abs().max().item()
print(f"wsplit S={S}: Δ={d:.4f}")
print("dbg head0 row0 d0-3:", o.view(B,S,NQ,D)[0,0,0,:4].tolist(), " row16:", o.view(B,S,NQ,D)[0,16,0,:4].tolist())
