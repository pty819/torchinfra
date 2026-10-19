# tests/test_paged_cache.py
import torch
from torchinfra.paged_cache import PagedKVCache


def main():
    torch.manual_seed(0)
    n_layers, n_kv, hd, block_size = 2, 4, 8, 4  # block=4 便于触发跨块
    cache = PagedKVCache(n_layers, n_kv, hd, num_blocks=8, block_size=block_size,
                         dtype=torch.float32)

    # 请求 0：两次 append 跨块（块大小 4：start=3 时 5 个 token 跨 3 块）
    req = 0
    n_blk = cache.alloc(req, n_tokens=3)
    assert n_blk == 1
    k1 = torch.randn(1, n_kv, 3, hd, device="cuda")
    v1 = torch.randn_like(k1)
    cache.append(req, 0, k1, v1, start=0)
    k2 = torch.randn(1, n_kv, 5, hd, device="cuda")
    v2 = torch.randn_like(k2)
    cache.append(req, 0, k2, v2, start=3)          # 占 3 块 → 共 4 块
    kg, vg = cache.gather(req, 0, upto=8)
    assert kg.shape == (1, n_kv, 8, hd)
    torch.testing.assert_close(kg, torch.cat([k1, k2], dim=2))
    torch.testing.assert_close(vg, torch.cat([v1, v2], dim=2))
    # 只读前缀
    k4, _ = cache.gather(req, 0, upto=4)
    torch.testing.assert_close(k4, torch.cat([k1, k2[:, :, :1]], dim=2))

    # 请求 1：独立槽位不串扰
    cache.alloc(1, n_tokens=5)
    ka = torch.randn(1, n_kv, 5, hd, device="cuda")
    cache.append(1, 0, ka, torch.zeros_like(ka), start=0)
    assert float((cache.gather(1, 0, 5)[0] - ka).abs().sum()) == 0.0

    # 第 1 层未被污染
    assert float(cache.k[1].abs().sum()) == 0.0

    # free 后复用且不残留旧数据：释放请求 0，再分配同规模必命中回收块
    used_before = set(cache.block_table[0][:4].tolist())
    cache.free(0)
    cache.alloc(2, n_tokens=16)
    used_after = set(cache.block_table[2].tolist())
    assert used_after == used_before
    k3 = torch.full((1, n_kv, 16, hd), 7.0, device="cuda")
    cache.append(2, 0, k3, k3, start=0)
    assert float((cache.gather(2, 0, 16)[0] - 7.0).abs().sum()) == 0.0
    print("PagedKVCache OK")


if __name__ == "__main__":
    main()
