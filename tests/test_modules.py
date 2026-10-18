# tests/test_modules.py — 随任务逐步追加
import torch
from torchinfra.modules import RMSNorm


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


def main():
    test_rms_norm()


if __name__ == "__main__":
    main()
