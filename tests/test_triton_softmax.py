# tests/test_triton_softmax.py
import torch
from torchinfra.triton_kernels import triton_softmax


def main():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        for n in (128, 4096, 1000):
            x = torch.randn(64, n, device="cuda", dtype=dtype) * 3
            out = triton_softmax(x)
            ref = torch.softmax(x.float(), dim=-1)
            torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-3)
    # 数值稳定：大 logit
    x = torch.tensor([[1000.0, 0.0, -1000.0, 1e30]], device="cuda")
    out = triton_softmax(x)
    ref = torch.softmax(x.float(), -1)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-6)
    print("triton_softmax OK")


if __name__ == "__main__":
    main()
