# wsplit2 正确性：GQA/MHA × S 整/非整 32 × 垃圾 pad（含 NaN 位模式）
# 输出清零防 allocator 复用假阳性；S>4096 参考逐 KV 组分块防 OOM
import sys
sys.path.insert(0, "~/torchinfra")
import torch, torch.nn.functional as F
from flydsl_kernels.fp8_wrapper import quantize_tensorwise

D = 128


def pad_garbage(t8, S, S_pad):
    if S_pad > S:
        v = t8.view(torch.int8)
        junk = torch.randint(-128, 128, tuple(v[:, :, S:, :].shape), device=v.device, dtype=torch.int8)
        junk.view(-1)[::97] = 127   # e4m3 NaN
        junk.view(-1)[::89] = -1    # e4m3 -NaN
        v[:, :, S:, :] = junk


def flydsl_wsplit2(q8, k8, v8, sq, sk, sv, nkv):
    import math
    from flydsl_kernels.flash_attn_fp8_wsplit2 import build_flash_attn_fp8_wsplit2_module
    B, NQ, S, D_ = q8.shape
    S_pad = (S + 127) // 128 * 128
    pad = S_pad - S
    ic8 = torch.int8
    if pad:
        q_p = F.pad(q8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        k_p = F.pad(k8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        v_p = F.pad(v8.view(ic8), (0, 0, 0, pad, 0, 0)).view(torch.float8_e4m3fn)
        pad_garbage(k_p, S, S_pad); pad_garbage(v_p, S, S_pad)
    else:
        q_p, k_p, v_p = q8, k8, v8
    o_p = torch.zeros(B, S_pad, NQ * D_, dtype=torch.bfloat16, device=q8.device)
    scl = torch.tensor([sq * sk / math.sqrt(D_) * math.log2(math.e), sv],
                       dtype=torch.float32, device=q8.device)
    exe = build_flash_attn_fp8_wsplit2_module(num_heads=NQ, head_dim=D_, num_kv_heads=nkv)
    exe(q_p.view(-1).view(ic8), k_p.view(-1).view(ic8), v_p.view(-1).view(ic8),
        o_p.view(-1), B, S, S_pad, scl)
    return o_p[:, :S].contiguous() if pad else o_p


def chunk_ref(q8, k8, v8, sq, sk, sv):
    """S>4096 逐 kv 组分块参考"""
    B, NQ, S, _ = q8.shape
    qf = q8.float() * sq
    parts = []
    if NQ == k8.shape[1]:
        for h in range(NQ):
            parts.append(F.scaled_dot_product_attention(qf[:, h:h + 1], k8[:, h:h + 1].float() * sk,
                                                        v8[:, h:h + 1].float() * sv))
    else:
        r = NQ // k8.shape[1]
        for g in range(k8.shape[1]):
            parts.append(F.scaled_dot_product_attention(qf[:, g * r:(g + 1) * r],
                                                        k8[:, g:g + 1].float() * sk,
                                                        v8[:, g:g + 1].float() * sv))
    return torch.cat(parts, dim=1)


def make(B, NQ, NKV, S, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    q = (torch.randn(B, NQ, S, D, device='cuda', generator=g) * 0.3).bfloat16()
    k = (torch.randn(B, NKV, S, D, device='cuda', generator=g) * 0.3).bfloat16()
    v = (torch.randn(B, NKV, S, D, device='cuda', generator=g) * 0.3).bfloat16()
    return q, k, v


torch.manual_seed(0)
print("== wsplit2 数值正确性（垃圾 pad，vs fp32 SDPA）==")
for (NQ, NKV), Ss in [((32, 8), [512, 544, 2048, 7102]), ((32, 32), [512, 544, 7102])]:
    for S in Ss:
        q, k, v = make(1, NQ, NKV, S, seed=S)
        q8, sq = quantize_tensorwise(q); k8, sk = quantize_tensorwise(k); v8, sv = quantize_tensorwise(v)
        ref = F.scaled_dot_product_attention(q8.float() * sq, k8.float() * sk, v8.float() * sv,
                                             enable_gqa=(NQ != NKV)) if S <= 4096 else chunk_ref(q8, k8, v8, sq, sk, sv)
        o = flydsl_wsplit2(q8, k8, v8, sq, sk, sv, NKV)
        torch.cuda.synchronize()
        d = (o.view(1, S, NQ, D).permute(0, 2, 1, 3).float() - ref).abs().max().item()
        tag = "GQA" if NQ != NKV else "MHA"
        print(f"  {tag} {NQ}/{NKV} S={S}: Δ={d:.4f} {'OK' if d < 0.02 else 'FAIL'}")
        del q, k, v, q8, k8, v8, ref, o
        torch.cuda.empty_cache()

print("== 重复稳定性 (MHA 32/32 S=7102 x3) ==")
q, k, v = make(1, 32, 32, 7102, seed=7)
q8, sq = quantize_tensorwise(q); k8, sk = quantize_tensorwise(k); v8, sv = quantize_tensorwise(v)
ref = chunk_ref(q8, k8, v8, sq, sk, sv)
for r in range(3):
    o = flydsl_wsplit2(q8, k8, v8, sq, sk, sv, 32)
    torch.cuda.synchronize()
    d = (o.view(1, 7102, 32, D).permute(0, 2, 1, 3).float() - ref).abs().max().item()
    print(f"  run {r}: Δ={d:.4f}")
print("ALL DONE")
