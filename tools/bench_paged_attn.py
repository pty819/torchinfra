"""P3 kernel A/B：两阶段（FlashDecoding split-K）vs 单 kernel 循环版。"""
import torch
import triton
from torchinfra.paged_cache import PagedKVCache
from torchinfra.paged_attn import paged_decode_attention, paged_decode_attention_loop


def setup(B, kv_len, dtype=torch.bfloat16):
    torch.manual_seed(0)
    paged = PagedKVCache(1, 8, 128, num_blocks=triton.cdiv(B * kv_len, 16) + 8,
                         block_size=16, dtype=dtype, max_seqs=max(B, 8))
    for r in range(B):
        k = torch.randn(1, 8, kv_len, 128, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        paged.append(r, 0, k, v, start=0)
    view = paged.view().slice_batch(list(range(B)))
    q = torch.randn(B, 16, 1, 128, device="cuda", dtype=dtype)
    return q, view


def main():
    # 正确性交叉验证：两版互拍（bf16）
    for B, L in ((1, 300), (4, 128), (8, 1000)):
        q, view = setup(B, L)
        a = paged_decode_attention(q, view).float()
        b = paged_decode_attention_loop(q, view).float()
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)
    print("两版输出一致 ✓\n")

    print(f"| B | kv_len | 两阶段 ms | 循环版 ms | 胜者 |")
    print("|---|---|---|---|---|")
    for B in (1, 2, 4, 8, 16):
        for L in (256, 1024):
            q, view = setup(B, L)
            t1 = triton.testing.do_bench(lambda: paged_decode_attention(q, view))
            t2 = triton.testing.do_bench(lambda: paged_decode_attention_loop(q, view))
            win = "两阶段" if t1 < t2 else "循环"
            print(f"| {B} | {L} | {t1:.3f} | {t2:.3f} | {win} ({t1 / t2:.2f}x) |")


if __name__ == "__main__":
    main()
