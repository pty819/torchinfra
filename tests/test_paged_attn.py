# tests/test_paged_attn.py — P3 attention 相关测试（随任务追加）
import torch
from torchinfra.kv_cache import KVCache
from torchinfra.paged_cache import PagedKVCache, MirroredKVCache


def test_mirror():
    """prefill 镜像：写入临时 contiguous cache 的同时镜像进分页。"""
    torch.manual_seed(0)
    n_layers, n_kv, hd, bs = 2, 4, 8, 4
    paged = PagedKVCache(n_layers, n_kv, hd, num_blocks=16, block_size=bs,
                         dtype=torch.float32)
    for req, L in ((0, 6), (1, 3), (2, 9)):
        mirror = MirroredKVCache(paged, req_id=req, max_seq=L)
        for layer in range(n_layers):
            k = torch.randn(1, n_kv, L, hd, device="cuda")
            v = torch.randn_like(k)
            mirror.append(layer, k, v, start=0)
            # 两边一致
            kc, vc = mirror.read(layer, L)
            kg, vg = paged.gather(req, layer, L)
            torch.testing.assert_close(kg, kc)
            torch.testing.assert_close(vg, vc)
        assert int(paged.kv_lens[req]) == L
    print("MirroredKVCache OK")


def main():
    test_mirror()


if __name__ == "__main__":
    main()
