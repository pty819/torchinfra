# tests/test_triton_rmsnorm.py
import torch
from torchinfra.modules import RMSNorm
from torchinfra.triton_kernels import TritonRMSNorm, fused_rmsnorm_residual


def test_rmsnorm():
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


def test_fused():
    torch.manual_seed(1)
    for dtype in (torch.bfloat16, torch.float32):
        n = 1024
        x = torch.randn(4, 65, n, device="cuda", dtype=dtype)
        dx = torch.randn_like(x)
        w = torch.randn(n, device="cuda", dtype=dtype)
        # torch 参考：add（存 dtype）→ RMSNorm（fp32 内算）
        res_ref = x + dx
        norm_ref = RMSNorm(n).cuda().to(dtype)
        with torch.no_grad():
            norm_ref.weight.copy_(w)
            nref = norm_ref(res_ref)
        res, normed = fused_rmsnorm_residual(x.contiguous(), dx.contiguous(),
                                             w, 1e-6)
        torch.testing.assert_close(res.float(), res_ref.float(),
                                   rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(normed.float(), nref.float(),
                                   rtol=1e-2, atol=1e-2)
    print("fused_rmsnorm_residual OK")


def main():
    test_rmsnorm()
    test_fused()


if __name__ == "__main__":
    main()
