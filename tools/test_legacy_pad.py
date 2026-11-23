import sys, math
import torch, torch.nn.functional as F
from flydsl_kernels.fp8_wrapper import quantize_tensorwise
from flydsl_kernels.flash_attn_fp8_gfx1201 import build_flash_attn_fp8_module
D = 128
legacy = build_flash_attn_fp8_module(num_heads=32, head_dim=D, num_kv_heads=32,
                                     causal=False, waves_per_eu=1, block_m=None,
                                     block_n=None, v_layout="transposed")
S, S_pad = 7102, 7168
g = torch.Generator(device='cuda').manual_seed(11)
q = (torch.randn(1, 32, S, D, device='cuda', generator=g) * 0.3).bfloat16()
k = (torch.randn(1, 32, S, D, device='cuda', generator=g) * 0.3).bfloat16()
v = (torch.randn(1, 32, S, D, device='cuda', generator=g) * 0.3).bfloat16()
q8, sq = quantize_tensorwise(q); k8, sk = quantize_tensorwise(k); v8, sv = quantize_tensorwise(v)
def head_ref(h):
    return F.scaled_dot_product_attention(q8[:, h:h+1].float()*sq, k8[:, h:h+1].float()*sk, v8[:, h:h+1].float()*sv)
ref = torch.cat([head_ref(h) for h in range(32)], dim=1)
scl = torch.tensor([sq*sk/math.sqrt(D)*math.log2(math.e), sv], dtype=torch.float32, device='cuda')
ic8 = torch.int8
def run(garbage):
    ps = []
    for t in (q8, k8, v8):
        p = F.pad(t.view(ic8), (0,0,0,S_pad-S,0,0)).view(torch.float8_e4m3fn)
        ps.append(p)
    if garbage:
        for p in ps[1:]:
            junk = torch.randint(-128, 128, tuple(p[:,:,S:,:].shape), device='cuda', dtype=torch.int8)
            junk.view(-1)[::97] = 127
            p.view(torch.int8)[:,:,S:,:] = junk
    o = torch.zeros(1, S_pad, 32*D, dtype=torch.bfloat16, device='cuda')
    legacy(ps[0].view(-1).view(ic8), ps[1].view(-1).view(ic8), ps[2].view(-1).view(ic8),
           o.view(-1), 1, S, S_pad, scl)
    return o[:, :S]
oz = run(False); oj = run(True)
torch.cuda.synchronize()
dz = (oz.view(1,S,32,D).permute(0,2,1,3).float()-ref).abs()
dj = (oj.view(1,S,32,D).permute(0,2,1,3).float()-ref).abs()
print(f"legacy 零pad: Δmax={dz.max():.4f} Δmean={dz.mean():.6f}")
print(f"legacy 垃圾pad: Δmax={dj.max():.4f} Δmean={dj.mean():.6f}")
print(f"两口味互差: {(oz.float()-oj.float()).abs().max():.4f}")
print(f"结论: {'零pad下也有~'+f'{dz.mean():.4f}'+' 系统性偏差 = pad 污染实锤' if dz.mean() > 0.001 else '零pad 无明显污染'}")
