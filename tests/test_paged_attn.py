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


# tests/test_paged_attn.py — P3 attention 相关测试（随任务追加）
import torch
from torchinfra.kv_cache import KVCache
from torchinfra.paged_cache import PagedKVCache, MirroredKVCache
from torchinfra.paged_attn import paged_decode_attention


def _ref_decode_attn(q, k, v, scale):
    """独立参考：gather 出的连续 K/V 上跑显式公式。q [H,1,D] or [B,H,1,D] 单请求。"""
    H, _, D = q.shape[0], q.shape[1], q.shape[-1]
    n_kv = k.shape[1]
    kc = k.repeat_interleave(H // n_kv, dim=1)   # [1,H,L,D]
    vc = v.repeat_interleave(H // n_kv, dim=1)
    qk = (q.float() @ kc[0].float().transpose(-1, -2)) * scale  # [H,1,L]
    p = torch.softmax(qk, dim=-1)
    return (p.to(v.dtype) @ vc[0]).squeeze(1)    # [H, D]


def test_paged_decode():
    torch.manual_seed(0)
    n_layers, n_kv, hd, bs = 1, 8, 128, 16
    H, D = 16, hd
    paged = PagedKVCache(n_layers, n_kv, hd, num_blocks=32, block_size=bs,
                         dtype=torch.float32)
    lens = [5, 33, 16]        # 含非整块尾 / 恰好整块
    reqs = list(range(len(lens)))
    ks, vs = {}, {}
    for r, L in zip(reqs, lens):
        k = torch.randn(1, n_kv, L, hd, device="cuda")
        v = torch.randn_like(k)
        paged.append(r, 0, k, v, start=0)      # append 自动 alloc/extend
        ks[r], vs[r] = k, v
    view = paged.view().slice_batch(reqs)

    q = torch.randn(len(reqs), H, 1, D, device="cuda")
    out = paged_decode_attention(q, view)
    assert out.shape == (len(reqs), H, D)
    scale = D ** -0.5
    for i, r in enumerate(reqs):
        ref = _ref_decode_attn(q[i], ks[r], vs[r], scale)
        torch.testing.assert_close(out[i].float(), ref.float(),
                                   rtol=1e-3, atol=1e-3)
    print("paged_decode_attention OK")


def main():
    test_mirror()
    test_paged_decode()


if __name__ == "__main__":
    main()
