# tests/test_batch_model.py
import torch
from torchinfra.triton_kernels import TritonRoPE
from torchinfra.batch_model import rope_decode_batch


def main():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        ref = TritonRoPE(128, 512, 1e6).cuda()
        B, Hq, Hk, D = 4, 16, 8, 128
        starts = torch.tensor([7, 0, 33, 1], dtype=torch.int32, device="cuda")
        q = torch.randn(B, Hq, 1, D, device="cuda", dtype=dtype)
        k = torch.randn(B, Hk, 1, D, device="cuda", dtype=dtype)
        q2, k2 = q.clone(), k.clone()
        rope_decode_batch(q2, k2, starts, ref.cos, ref.sin)
        for b in range(B):
            qr, kr = ref.apply_qk(q[b:b + 1].clone(), k[b:b + 1].clone(),
                                  int(starts[b]))
            torch.testing.assert_close(q2[b].float(), qr[0].float(),
                                       rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(k2[b].float(), kr[0].float(),
                                       rtol=1e-2, atol=1e-2)
    print("rope_decode_batch OK")


if __name__ == "__main__":
    main()
