# tests/test_triton_rmsnorm.py
import torch
from torchinfra.modules import RMSNorm
from torchinfra.triton_kernels import TritonRMSNorm


def main():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        for n in (1024, 1000):
            x = torch.randn(8, 33, n, device="cuda", dtype=dtype)
            ref_m = RMSNorm(n).cuda().to(dtype)
            tri_m = TritonRMSNorm(n).cuda().to(dtype)
            with torch.no_grad():
                tri_m.weight.copy_(ref_m.weight)
            with torch.no_grad():
                a, b = ref_m(x), tri_m(x)
            torch.testing.assert_close(a.float(), b.float(), rtol=1e-2, atol=1e-2)
            assert b.dtype == dtype
    print("TritonRMSNorm OK")


if __name__ == "__main__":
    main()
