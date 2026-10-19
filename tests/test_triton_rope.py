# tests/test_triton_rope.py
import torch
from torchinfra.modules import RotaryEmbedding
from torchinfra.triton_kernels import TritonRoPE


def main():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        ref = RotaryEmbedding(128, 512, 1e6).cuda()
        tri = TritonRoPE(128, 512, 1e6).cuda()
        q = torch.randn(1, 16, 9, 128, device="cuda", dtype=dtype)
        k = torch.randn(1, 8, 9, 128, device="cuda", dtype=dtype)
        q2, k2 = q.clone(), k.clone()
        qr_ref, kr_ref = ref.apply_qk(q.clone(), k.clone(), 7)
        qr_tri, kr_tri = tri.apply_qk(q2, k2, 7)
        torch.testing.assert_close(qr_tri.float(), qr_ref.float(),
                                   rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(kr_tri.float(), kr_ref.float(),
                                   rtol=1e-2, atol=1e-2)
    print("TritonRoPE OK")


if __name__ == "__main__":
    main()
