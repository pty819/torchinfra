"""decode 路径投影融合：QKV 三合一 + gate/up 二合一（vLLM 标准做法）。

瘦 GEMM（B=8 时每个 1024×2048 级别）带宽利用率低，纵拼后 kernel 数从
每层 6 个降到 2 个。原 Linear 保留（prefill 继续用），fused 权重是额外
~30% 显存换 decode 带宽。幂等；不可用时自动回退原路径。
"""
import torch
import torch.nn.functional as F


def fuse_decode_projections(model) -> None:
    for layer in model.model.layers:
        a = layer.self_attn
        if not hasattr(a, "w_qkv"):
            a.w_qkv = torch.cat([a.q_proj.weight.data, a.k_proj.weight.data,
                                 a.v_proj.weight.data], dim=0).contiguous()
        m = layer.mlp
        if not hasattr(m, "w_gate_up"):
            m.w_gate_up = torch.cat([m.gate_proj.weight.data,
                                     m.up_proj.weight.data],
                                    dim=0).contiguous()


def attn_qkv(h: torch.Tensor, attn):
    """h [B,1,hidden] → (q, k, v)，各 [B,1,heads,hd]（调用方自行 transpose）。"""
    if hasattr(attn, "w_qkv"):
        qkv = F.linear(h, attn.w_qkv)
        nq = attn.n_heads * attn.hd
        nk = attn.n_kv * attn.hd
        q = qkv[..., :nq].view(-1, 1, attn.n_heads, attn.hd)
        k = qkv[..., nq:nq + nk].view(-1, 1, attn.n_kv, attn.hd)
        v = qkv[..., nq + nk:].view(-1, 1, attn.n_kv, attn.hd)
        return q, k, v
    b = h.shape[0]
    return (attn.q_proj(h).view(b, 1, attn.n_heads, attn.hd),
            attn.k_proj(h).view(b, 1, attn.n_kv, attn.hd),
            attn.v_proj(h).view(b, 1, attn.n_kv, attn.hd))


def mlp_forward(h: torch.Tensor, mlp) -> torch.Tensor:
    if hasattr(mlp, "w_gate_up"):
        gate, up = F.linear(h, mlp.w_gate_up).chunk(2, dim=-1)
        return mlp.down_proj(F.silu(gate) * up)
    return mlp(h)
