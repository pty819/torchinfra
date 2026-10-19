# tests/test_triton_add.py
import torch
from torchinfra.triton_kernels import triton_add


def main():
    for dtype in (torch.bfloat16, torch.float32):
        for n in (1 << 20, 1_000_003):
            x = torch.randn(n, device="cuda", dtype=dtype)
            y = torch.randn_like(x)
            out = triton_add(x, y)
            torch.cuda.synchronize()
            torch.testing.assert_close(out.float(), (x.float() + y.float()),
                                       rtol=1e-2, atol=1e-2)
    print("triton_add OK")


if __name__ == "__main__":
    main()
