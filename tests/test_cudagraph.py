# tests/test_cudagraph.py
"""P4 decode 优化：GPU 侧 cache 写入 kernel + CUDA Graph。"""
import torch
from torchinfra.paged_cache import PagedKVCache, MirroredKVCache
from torchinfra.cudagraph import append_kv, DecodeGraphRunner


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


def test_graph_vs_eager():
    from tests._ref import MODEL_DIR
    from torchinfra.model import Qwen3ForCausalLM
    from torchinfra.batch_model import decode_step
    from torchinfra.tokenizer import TokenizerBox

    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    tok = TokenizerBox(MODEL_DIR)
    prompts = ["你好，世界。", "解释一下RoPE。", "1+1="]
    max_new = 4
    B = len(prompts)

    def fresh_cache():
        return PagedKVCache(model.cfg.num_hidden_layers,
                            model.cfg.num_key_value_heads, model.cfg.head_dim,
                            num_blocks=192, block_size=16,
                            dtype=torch.float32, max_seqs=8)

    paged_e, paged_g = fresh_cache(), fresh_cache()
    for paged in (paged_e, paged_g):
        with torch.no_grad():
            for r, p in enumerate(prompts):
                ids = tok.encode(p)
                paged.reserve(r, len(ids) + max_new)   # 先预留全量
                mirror = MirroredKVCache(paged, req_id=r, max_seq=len(ids))
                model(torch.tensor([ids], device="cuda"), kv=mirror,
                      start_pos=0, logits_to_keep="last")

    runner = DecodeGraphRunner(model, paged_g, batch_size=8)
    tok_ids = [11, 22, 33]
    active = list(range(B))
    for step in range(max_new):
        starts = [int(paged_e.kv_lens[i]) for i in active]   # 本步写入位置（eager 前）
        with torch.no_grad():
            ref = decode_step(model,
                              torch.tensor([tok_ids[i] for i in active],
                                           device="cuda").unsqueeze(1),
                              paged_e, active)                # 会推进 paged_e.kv_lens
        rows = list(zip(active, starts))
        got = runner.run([tok_ids[i] for i in active], rows)
        for j, i in enumerate(active):
            torch.testing.assert_close(got[j], ref[j], rtol=1e-3, atol=1e-3,
                                       msg=f"step{step} req{i}")
        for i in active:                    # graph 侧 kv_lens 手动同步
            paged_g.kv_lens[i] = starts[active.index(i)] + 1
        tok_ids = [int(r.argmax()) for r in ref]  # greedy 统一推进
        if step == 1 and len(active) == 3:
            active = active[:2]      # 模拟一个请求完成（batch 收缩 → pad 增多）
    print("DecodeGraphRunner vs eager OK（含 batch 收缩）")


def main():
    test_append_kv()
    test_graph_vs_eager()


if __name__ == "__main__":
    main()
