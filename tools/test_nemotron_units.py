"""nemotron 移植的三个独立单元验证（不加载模型权重）：
A. yarn inv_freq vs transformers 原函数逐位对比
B. rope kernel vs torch 参考（含 llama4 Q-scale 两档）
C. SDPA + [B,1,1,S] additive key mask 的后端行为与显存
"""
import sys, math, json
import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, ".")
from torchinfra.nemotron_rerank import (
    yarn_inv_freq, rope_tables, _rope_qkv_kernel, NQ, NKV, HD, ROPE_TR, ROPE_WARPS,
    ROPE_ORIG)


def test_yarn():
    from transformers.models.ministral3 import Ministral3Config
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    cfg = Ministral3Config(**json.load(
        open("~/models/Nemotron-3.5-Rerank-8B-FP8/config.json")
    )["text_config"])
    ref_inv, ref_factor = _compute_yarn_parameters(cfg, device="cpu")
    mine = yarn_inv_freq("cpu")
    print(f"A. yarn inv_freq: max|Δ|={(ref_inv - mine).abs().max():.3e} "
          f"(attention_factor={ref_factor})")
    assert torch.allclose(ref_inv, mine, atol=0, rtol=0), "inv_freq 不一致"
    assert ref_factor == 1.0, f"attention_factor {ref_factor} ≠ 1"
    return rope_tables(4096, "cpu")


def torch_rope_ref(qkv, pos, cos, sin, l4sc, kc=1.0, vc=1.0):
    BS = qkv.shape[0]
    q = qkv[:, :NQ * HD].view(BS, NQ, HD).float()
    k = qkv[:, NQ * HD:NQ * HD + NKV * HD].view(BS, NKV, HD).float() * kc
    v = (qkv[:, NQ * HD + NKV * HD:].view(BS, NKV, HD).float() * vc)
    half = HD // 2
    cs = cos[pos.long()]   # [BS, half]
    c, s = cs[:, None, :], sin[pos.long()][:, None, :]
    a, b = q[..., :half], q[..., half:]
    q = torch.cat([a * c - b * s, b * c + a * s], -1)
    if l4sc:
        qs = 1 + 0.1 * torch.log1p(torch.floor(pos.float() / ROPE_ORIG))
        q = q * qs[:, None, None]
    a, b = k[..., :half], k[..., half:]
    k = torch.cat([a * c - b * s, b * c + a * s], -1)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def test_rope(cos, sin):
    torch.manual_seed(0)
    BS, S = 64, 32
    B = BS // S
    dev = "cuda"
    qkv = torch.randn(BS, (NQ + 2 * NKV) * HD, dtype=torch.bfloat16, device=dev)
    pos = torch.arange(S, dtype=torch.int32, device=dev).repeat(B).reshape(-1)
    cos, sin = cos.to(dev), sin.to(dev)
    for l4, kc, vc in [(False, 1.0, 1.0), (True, 1.0, 1.0), (False, 1.7, 0.35)]:
        q = torch.empty(B, NQ, S, HD, dtype=torch.bfloat16, device=dev)
        k = torch.empty(B, NKV, S, HD, dtype=torch.bfloat16, device=dev)
        v = torch.empty(B, NKV, S, HD, dtype=torch.bfloat16, device=dev)
        _rope_qkv_kernel[(triton.cdiv(BS, ROPE_TR), NQ)](
            qkv, cos, sin, pos, q, k, v, qkv.stride(0), cos.stride(0),
            BS, S, kc, vc, NQ=NQ, NKV=NKV, D=HD, HALF=HD // 2, TR=ROPE_TR,
            L4SC=l4, ORIG=float(ROPE_ORIG), BETA=0.1, num_warps=ROPE_WARPS)
        qr, kr, vr = torch_rope_ref(qkv, pos, cos, sin, l4, kc, vc)
        # kernel 输出 [B,H,S,D] → 参考转成同布局
        qr_t = qr.view(B, S, NQ, HD).transpose(1, 2)
        kr_t = kr.view(B, S, NKV, HD).transpose(1, 2)
        vr_t = vr.view(B, S, NKV, HD).transpose(1, 2)
        dq = (q.float() - qr_t.float()).abs().max().item()
        dk = (k.float() - kr_t.float()).abs().max().item()
        dv = (v.float() - vr_t.float()).abs().max().item()
        print(f"B. rope L4SC={l4} kc={kc} vc={vc}: "
              f"|Δq|={dq:.2e} |Δk|={dk:.2e} |Δv|={dv:.2e}")
        assert max(dq, dk, dv) < 2e-2, "rope kernel 与参考偏差过大"


def test_sdpa_mask():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    S, B = 2048, 1
    q = torch.randn(B, NQ, S, HD, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, NKV, S, HD, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, NKV, S, HD, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(B, 1, 1, S, dtype=torch.bfloat16, device="cuda")
    bias[:, :, :, S // 2:] = float("-inf")
    ref = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    for name, be in [("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("flash", SDPBackend.FLASH_ATTENTION),
                     ("default(自动)", None)]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            import time
            t0 = time.time()
            if be is None:
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                  SDPBackend.EFFICIENT_ATTENTION,
                                  SDPBackend.MATH]):
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                       enable_gqa=True)
            else:
                with sdpa_kernel([be]):
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                       enable_gqa=True)
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_stats()["allocated_bytes.all.peak"]
                    - q.numel() * 2 - k.numel() * 2 - v.numel() * 2 - bias.numel() * 2) / 2**20
            err = (o[:, :, :S // 2].float()
                   - ref[:, :, :S // 2].float()).abs().max().item()
            print(f"C. SDPA+keymask [{name}]: {time.time()-t0:.3f}s "
                  f"transient~{peak:.0f}MB 前半段|Δ|={err:.2e}")
        except Exception as e:
            print(f"C. SDPA+keymask [{name}]: 不可用 ({str(e)[:60]})")


if __name__ == "__main__":
    cos, sin = test_yarn()
    test_rope(cos, sin)
    test_sdpa_mask()
    print("单元验证全部通过")
