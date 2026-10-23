"""滑窗 kernel v2（三段免掩码 + [B,S,H*D] 直写）的配置扫描与对拍。"""
import sys

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, ".")
from torchinfra.jina_rerank import _window_attn_kernel  # noqa: E402

H, KV, D, W = 16, 8, 128, 1024
SM = D ** -0.5


def reference(q, k, v, pad):
    """torch 参考实现：显式 window+causal mask SDPA。q [B,H,S,D]。"""
    B, _, S, _ = q.shape
    idx = torch.arange(S, device=q.device)
    causal = idx[None, None, :, None] >= idx[None, None, None, :]   # q >= k
    win = idx[None, None, :, None] - W < idx[None, None, None, :]   # q - W < k
    vq = idx[None, None, :, None] >= pad[:, None, None, None]       # [B,1,S,1]
    vk = idx[None, None, None, :] >= pad[:, None, None, None]       # [B,1,1,S]
    mask = causal & win & vq & vk                                   # [B,1,S,S]
    bias = torch.zeros_like(mask, dtype=q.dtype)
    bias.masked_fill_(~mask, float("-inf"))
    k2 = k.repeat_interleave(H // KV, dim=1)
    v2 = v.repeat_interleave(H // KV, dim=1)
    out = F.scaled_dot_product_attention(q, k2, v2, attn_mask=bias)
    return out.transpose(1, 2).reshape(B, S, H * D)


def run_new(q, k, v, pad, cfg):
    TQ, TK, warps, stages = cfg
    B, _, S, _ = q.shape
    out = torch.empty(B, S, H * D, dtype=q.dtype, device=q.device)
    _window_attn_kernel[(B, H, triton.cdiv(S, TQ))](
        q, k, v, out, pad, SM, S,
        NQ_PER_KV=H // KV, D=D, TQ=TQ, TK=TK, WINDOW=W,
        num_warps=warps, num_stages=stages)
    return out


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g0, g1 = torch.cuda.Event(True), torch.cuda.Event(True)
    g0.record()
    for _ in range(n):
        fn()
    g1.record()
    torch.cuda.synchronize()
    return g0.elapsed_time(g1) / n


def main():
    torch.manual_seed(0)
    cfgs = [(64, 64, 4, 1), (64, 64, 4, 2), (128, 64, 8, 2), (128, 64, 8, 3),
            (128, 64, 4, 2), (128, 128, 8, 2), (64, 64, 8, 2)]
    # 正确性：含左 padding、S 跨过 W 边界的组合
    for B, S, pad_len in [(1, 2048, 0), (2, 3000, 512), (1, 5000, 0),
                          (1, 8782, 0), (1, 27143, 0)]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
        k = torch.randn(B, KV, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
        v = torch.randn(B, KV, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
        pad = torch.full((B,), pad_len, dtype=torch.int32, device="cuda")
        ref = reference(q, k, v, pad)
        for cfg in cfgs:
            got = run_new(q, k, v, pad, cfg)
            # 只比有效行：pad 行两侧都是未定义值（kernel 不写、SDPA 全 -inf）
            idx = torch.arange(S, device="cuda")
            for bi in range(B):
                vq = idx >= int(pad[bi])
                err = (got[bi][vq].float() - ref[bi][vq].float()).abs().max().item()
                assert err < 0.02, f"B{B} S{S} pad{pad_len} cfg{cfg}: err {err}"
        print(f"correctness B={B} S={S} pad={pad_len}: OK", flush=True)
    # 性能
    for S in [8782, 27143]:
        q = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16)
        pad = torch.zeros(1, dtype=torch.int32, device="cuda")
        ref_ms = bench(lambda: reference(q, k, v, pad), n=3)
        print(f"S={S} SDPA-masked-reference {ref_ms:.2f} ms", flush=True)
        for cfg in cfgs:
            t = bench(lambda: run_new(q, k, v, pad, cfg))
            fl = 4 * S * W * H * D
            print(f"S={S} cfg={cfg}: {t:.2f} ms  {fl/(t/1e3)/1e12:.0f} TF",
                  flush=True)


if __name__ == "__main__":
    main()
