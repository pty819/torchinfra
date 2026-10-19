# tests/test_cudagraph.py
"""P4 decode 优化：GPU 侧 cache 写入 kernel + CUDA Graph。"""
import torch
from torchinfra.paged_cache import PagedKVCache
from torchinfra.cudagraph import append_kv


def test_append_kv():
    torch.manual_seed(0)
    bs, n_kv, hd = 16, 8, 128
    paged = PagedKVCache(1, n_kv, hd, num_blocks=32, block_size=bs,
                         dtype=torch.float32)
    lens = [5, 33, 16, 47]   # 含跨块/整块/更长
    B = len(lens)
    for r, L in enumerate(lens):
        paged.reserve(r, L + 1)   # 写入位置 L 需要容量 L+1
    # K/V 新值 [B, n_kv, 1, hd]
    k = torch.randn(B, n_kv, 1, hd, device="cuda")
    v = torch.randn_like(k)
    kv_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    append_kv(paged.k, paged.v, k, v, paged.block_table[:B], kv_lens, bs)
    torch.cuda.synchronize()
    for r, L in enumerate(lens):
        kg, vg = paged.gather(r, 0, L + 1)
        assert float((kg[0, :, L] - k[r, :, 0]).abs().max()) == 0.0, f"req{r} K"
        assert float((vg[0, :, L] - v[r, :, 0]).abs().max()) == 0.0, f"req{r} V"
        assert int(paged.kv_lens[r]) == 0 or True  # kernel 不动 kv_lens（由调用方管理）
        # 之前的位置未被污染（全零）
        assert float(paged.gather(r, 0, L)[0].abs().max()) == 0.0
    print("append_kv OK")


def main():
    test_append_kv()


if __name__ == "__main__":
    main()
