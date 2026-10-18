# tests/test_modules.py — 随任务逐步追加
import math
import torch
from torchinfra.modules import RMSNorm, RotaryEmbedding, rotate_half


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


def main():
    test_rms_norm()
    test_rope()


if __name__ == "__main__":
    main()
