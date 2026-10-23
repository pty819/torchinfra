"""qk_norm_rope(+v搬运) kernel 的配置扫描与对拍（vs torch 参考）。"""
import sys

import torch
import triton

sys.path.insert(0, ".")
from torchinfra.jina_rerank import _qk_norm_rope_kernel  # noqa: E402

NQ, NKV, D = 16, 8, 128
HALF = D // 2
EPS = 1e-6


def torch_ref(q_in, k_in, v_in, qnw, knw, cos, sin, pos, S):
    n = q_in.shape[0]
    q = q_in.float()
    a, b = q[..., :HALF], q[..., HALF:]
    rms = torch.rsqrt(((a * a).sum(-1) + (b * b).sum(-1)) / D + EPS)
    a = a * rms[..., None] * qnw.float()[None, :HALF]
    b = b * rms[..., None] * qnw.float()[None, HALF:]
    c = cos[pos].float()[:, :HALF][:, None, :]
    s = sin[pos].float()[:, :HALF][:, None, :]
    qo = torch.cat([a * c - b * s, b * c + a * s], -1).to(torch.bfloat16)
    k = k_in.float()
    a, b = k[..., :HALF], k[..., HALF:]
    rms = torch.rsqrt(((a * a).sum(-1) + (b * b).sum(-1)) / D + EPS)
    a = a * rms[..., None] * knw.float()[None, :HALF]
    b = b * rms[..., None] * knw.float()[None, HALF:]
    ko = torch.cat([a * c - b * s, b * c + a * s], -1).to(torch.bfloat16)
    return qo.permute(1, 0, 2), ko.permute(1, 0, 2)  # [NQ,S,D], [NKV,S,D]


def run(q_in, k_in, v_in, qnw, knw, cos, sin, pos, S, tr, warps):
    n = q_in.shape[0]
    q = torch.empty(1, NQ, S, D, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(1, NKV, S, D, dtype=torch.bfloat16, device="cuda")
    v = torch.empty(1, NKV, S, D, dtype=torch.bfloat16, device="cuda")
    _qk_norm_rope_kernel[(triton.cdiv(n, tr), NQ)](
        q_in, k_in, v_in, qnw, knw, cos, sin, pos, q, k, v,
        q_in.stride(0), k_in.stride(0), v_in.stride(0), cos.stride(0),
        EPS, n, S, NQ=NQ, NKV=NKV, D=D, HALF=HALF, TR=tr, WANT_V=True,
        num_warps=warps)
    return q, k, v


def bench(fn, n=30):
    for _ in range(5):
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
    for S in [8782, 27143]:
        qkv = torch.randn(S, (NQ + 2 * NKV) * D, device="cuda",
                          dtype=torch.bfloat16) * 0.3
        q_in = qkv[:, :NQ * D].view(S, NQ, D)
        k_in = qkv[:, NQ * D:(NQ + NKV) * D].view(S, NKV, D)
        v_in = qkv[:, (NQ + NKV) * D:].view(S, NKV, D)
        qnw = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        knw = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(S, device="cuda", dtype=torch.int32)
        inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, dtype=torch.float32,
                                          device="cuda") / D))
        t = torch.arange(S, dtype=torch.float32, device="cuda")
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), -1)
        cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
        rq, rk = torch_ref(q_in, k_in, v_in, qnw, knw, cos, sin, pos, S)
        rv = v_in.permute(1, 0, 2)
        for tr, warps in [(16, 4), (16, 8), (32, 4), (32, 8), (64, 8)]:
            q, k, v = run(q_in, k_in, v_in, qnw, knw, cos, sin, pos, S, tr, warps)
            eq = ((q[0].float() - rq.float()).abs().max().item(),
                  (k[0].float() - rk.float()).abs().max().item(),
                  (v[0].float() - rv.float()).abs().max().item())
            assert max(eq) < 0.05, (tr, warps, eq)  # bf16 单 ulp@幅值4 ≈ 0.03
            t_kr = bench(lambda: run(q_in, k_in, v_in, qnw, knw, cos, sin,
                                     pos, S, tr, warps))
            # 旧路径对照：kernel(WANT_V=False 等价于 v 不搬) + 单独 permute
            t_old = bench(lambda: v_in.view(1, S, NKV, D)
                          .permute(0, 2, 1, 3).contiguous())
            gb = (S * (NQ + 2 * NKV) * D * 2 + S * (NQ + 2 * NKV) * D * 2) / 1e9
            print(f"S={S} TR={tr} w={warps}: fused {t_kr:.3f} ms "
                  f"({gb/t_kr*1e3:.0f} GB/s) | 单独 permute {t_old:.3f} ms",
                  flush=True)


if __name__ == "__main__":
    main()
