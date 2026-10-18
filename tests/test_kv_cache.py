# tests/test_kv_cache.py
import torch
from torchinfra.kv_cache import KVCache


def main():
    torch.manual_seed(0)
    kv = KVCache(n_layers=2, max_seq=16, n_kv_heads=4, head_dim=8,
                 dtype=torch.bfloat16, device="cuda")
    k1 = torch.randn(1, 4, 5, 8, device="cuda", dtype=torch.bfloat16)
    v1 = torch.randn_like(k1)
    kv.append(0, k1, v1, start=0)
    k2 = torch.randn(1, 4, 3, 8, device="cuda", dtype=torch.bfloat16)
    v2 = torch.randn_like(k2)
    kv.append(0, k2, v2, start=5)
    k, v = kv.read(0, upto=8)
    assert k.shape == (1, 4, 8, 8)
    ref = torch.cat([k1, k2], dim=2)  # [1, nkv, 8, d] 即期望布局
    torch.testing.assert_close(k, ref)
    torch.testing.assert_close(v, torch.cat([v1, v2], dim=2))
    # 只读前缀
    k5, _ = kv.read(0, upto=5)
    torch.testing.assert_close(k5, k1)
    # 第 1 层未被污染
    assert float(kv.k[1].abs().sum()) == 0.0
    print("KVCache OK")


if __name__ == "__main__":
    main()
