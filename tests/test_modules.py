# tests/test_modules.py — 随任务逐步追加
import math
import torch
from torchinfra.kv_cache import KVCache
from torchinfra.modules import RMSNorm, RotaryEmbedding, Attention, rotate_half


def test_rms_norm():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(2, 17, 64, device="cuda", dtype=dtype)
        m = RMSNorm(64, eps=1e-6).cuda().to(dtype)
        with torch.no_grad():
            m.weight.uniform_(0.5, 1.5)
        # 独立参考：全 fp32 手写公式
        x32 = x.float()
        ref = m.weight.float() * (
            x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6))
        out = m(x)
        torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
        assert out.dtype == dtype
    print("RMSNorm OK")


def test_rope():
    # rotate_half 基本性质
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda")
    assert torch.equal(rotate_half(x),
                       torch.tensor([[-3.0, -4.0, 1.0, 2.0]], device="cuda"))

    # 手推解析解：head_dim=4，theta=10
    # inv_freq = [1.0, rho]，rho = 10**(-2/4)
    hd, theta, max_pos = 4, 10.0, 64
    rope = RotaryEmbedding(hd, max_pos, theta, device="cuda")
    q = torch.randn(1, 2, 5, hd, device="cuda")
    k = torch.randn(1, 3, 5, hd, device="cuda")
    qr, kr = rope.apply_qk(q, k, start=7)
    rho = theta ** (-2.0 / hd)
    for p_i, p in enumerate(range(7, 12)):
        c, s = math.cos(p), math.sin(p)
        c2, s2 = math.cos(p * rho), math.sin(p * rho)
        exp = torch.stack([  # half-split 旋转公式的期望值
            q[0, 0, p_i, 0] * c - q[0, 0, p_i, 2] * s,
            q[0, 0, p_i, 1] * c2 - q[0, 0, p_i, 3] * s2,
            q[0, 0, p_i, 2] * c + q[0, 0, p_i, 0] * s,
            q[0, 0, p_i, 3] * c2 + q[0, 0, p_i, 1] * s2,
        ])
        torch.testing.assert_close(qr[0, 0, p_i].float().cpu(), exp.float().cpu(),
                                   rtol=1e-3, atol=1e-3)
    print("RoPE OK")


def _ref_attention(hidden, attn, rope):
    """独立参考：显式 eager 公式（复用 attn 的权重，不复用其 forward）。"""
    b, s, _ = hidden.shape
    n_h, n_kv, hd = attn.n_heads, attn.n_kv, attn.hd
    q = attn.q_norm(attn.q_proj(hidden).view(b, s, n_h, hd).transpose(1, 2))
    k = attn.k_norm(attn.k_proj(hidden).view(b, s, n_kv, hd).transpose(1, 2))
    v = attn.v_proj(hidden).view(b, s, n_kv, hd).transpose(1, 2)
    q, k = rope.apply_qk(q, k, start=0)
    k = k.repeat_interleave(n_h // n_kv, dim=1)
    v = v.repeat_interleave(n_h // n_kv, dim=1)
    sc = (q.float() @ k.float().transpose(-1, -2)) * attn.scale
    mask = torch.arange(s, device="cuda")[None, :] > \
           torch.arange(s, device="cuda")[:, None]
    sc = sc.masked_fill(mask[None, None], float("-inf"))
    out = torch.softmax(sc, dim=-1).to(v.dtype) @ v
    return attn.o_proj(out.transpose(1, 2).reshape(b, s, n_h * hd))


def test_attention():
    from torchinfra.config import Qwen3Config
    cfg = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=4, head_dim=16, rms_norm_eps=1e-6,
                      rope_theta=10000.0, max_position_embeddings=64,
                      tie_word_embeddings=False)
    torch.manual_seed(1)
    attn = Attention(cfg).cuda().to(torch.bfloat16).eval()
    rope = RotaryEmbedding(cfg.head_dim, 64, cfg.rope_theta).cuda()

    x = torch.randn(1, 9, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    # 用例 A：一次算整段 == 独立参考
    kv = KVCache(1, 64, cfg.num_key_value_heads, cfg.head_dim)
    with torch.no_grad():
        out_full = attn(x, rope, kv, layer_idx=0, start_pos=0)
        out_ref = _ref_attention(x, attn, rope)
    torch.testing.assert_close(out_full.float(), out_ref.float(),
                               rtol=1e-2, atol=1e-2)
    # 用例 B：分块（prefill 5 + 续算 4）与整段一致 —— cache 语义正确性
    kv2 = KVCache(1, 64, cfg.num_key_value_heads, cfg.head_dim)
    with torch.no_grad():
        attn(x[:, :5], rope, kv2, 0, start_pos=0)
        out_chunk = attn(x[:, 5:], rope, kv2, 0, start_pos=5)
    torch.testing.assert_close(out_chunk.float(), out_full[:, 5:].float(),
                               rtol=1e-2, atol=1e-2)
    print("Attention OK")


def main():
    test_rms_norm()
    test_rope()
    test_attention()


if __name__ == "__main__":
    main()
