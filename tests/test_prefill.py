"""批量 prefill 差分测试：打包前向 vs 逐请求 P1 前向（fp32）。

参考与被测完全独立：参考走 model.forward + KVCache（P1 路径），
被测走 prefill_batch（打包 GEMM + varlen kernel + 打包 cache 写入）。
长度特意含 16/17/33（块边界 ±1）抓块表寻址错位。
"""
import torch

from tests._ref import MODEL_DIR
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.kv_cache import KVCache
from torchinfra.paged_cache import PagedKVCache
from torchinfra.prefill import prefill_batch

# 1 / 块内 / 块边界±1 / 跨多块，ragged
LENS = [1, 5, 16, 17, 33, 64]


def main():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    cfg = model.cfg
    prompts = [torch.randint(1000, 50000, (L,)).tolist() for L in LENS]

    # ---- 参考串行：P1 路径逐请求 ----
    ref_logits = []
    ref_k = {}          # (req, layer) → (k, v) 各 [1, n_kv, S, D]
    for rid, p in enumerate(prompts):
        kv = KVCache(cfg.num_hidden_layers, len(p) + 8,
                     cfg.num_key_value_heads, cfg.head_dim,
                     dtype=torch.float32)
        with torch.no_grad():
            lg = model(torch.tensor([p], device="cuda"), kv=kv, start_pos=0,
                       logits_to_keep="last")
        ref_logits.append(lg[0, -1])
        ref_k[rid] = {i: kv.read(i, len(p)) for i in
                      range(cfg.num_hidden_layers)}

    # ---- 被测：打包批量 ----
    paged = PagedKVCache(cfg.num_hidden_layers, cfg.num_key_value_heads,
                         cfg.head_dim, num_blocks=64, block_size=16,
                         dtype=torch.float32)
    for rid, p in enumerate(prompts):
        paged.reserve(rid, len(p))
    with torch.no_grad():
        got = prefill_batch(model, paged, prompts, bt_rows=list(range(len(prompts))))

    assert got.shape == (len(prompts), cfg.vocab_size), got.shape
    worst = 0.0
    for i in range(len(prompts)):
        d = (got[i] - ref_logits[i]).abs().max().item()
        worst = max(worst, d)
        assert torch.equal(got[i].argmax(), ref_logits[i].argmax()), \
            f"argmax 不一致 @请求{i}（logit diff {d:.2e}）"
    print(f"logits 对拍 OK（max abs diff {worst:.2e}，argmax 全一致）")

    # ---- cache 内容对拍（取首中尾三层）----
    for i in (0, cfg.num_hidden_layers // 2, cfg.num_hidden_layers - 1):
        for rid, p in enumerate(prompts):
            S = len(p)
            gk, gv = paged.gather(rid, i, upto=S)
            rk, rv = ref_k[rid][i]
            dk = (gk - rk).abs().max().item()
            dv = (gv - rv).abs().max().item()
            assert dk < 2e-3 and dv < 2e-3, f"cache 偏差 layer={i} rid={rid}: k={dk} v={dv}"
    print("cache 写入对拍 OK（k/v 各层各请求 max diff < 2e-3）")
    print("prefill_batch OK")


if __name__ == "__main__":
    main()
