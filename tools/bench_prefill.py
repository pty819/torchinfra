"""长输入 prefill 基准：token 吞吐与 TFLOPS 随长度/批量的变化。

用法：PYTHONPATH=. python tools/bench_prefill.py
"""
import random
import torch

from tests._ref import MODEL_DIR
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.paged_cache import PagedKVCache
from torchinfra.prefill import prefill_batch

# 真实场景代表性的 (B, S) 组合：短问答 / 中等 / 长文档
CASES = [(1, 128), (8, 128), (1, 1024), (4, 1024), (1, 2048), (2, 2048),
         (1, 4096), (1, 8192)]
GFLOP_PER_TOKEN = 0.88      # 2 × ~440M 非 embedding 计算参数


def main():
    random.seed(0)
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR)
    cfg = model.cfg
    for B, S in CASES:
        prompts = [[random.randint(1000, 50000) for _ in range(S)]
                   for _ in range(B)]
        pg = PagedKVCache(cfg.num_hidden_layers, cfg.num_key_value_heads,
                          cfg.head_dim, num_blocks=B * (S + 16) // 16 + 16,
                          block_size=16)
        for r in range(B):
            pg.reserve(r, S)
        rows = list(range(B))
        with torch.no_grad():
            for _ in range(3):
                prefill_batch(model, pg, prompts, bt_rows=rows)
            torch.cuda.synchronize()
            g0 = torch.cuda.Event(enable_timing=True)
            g1 = torch.cuda.Event(enable_timing=True)
            g0.record()
            for _ in range(5):
                prefill_batch(model, pg, prompts, bt_rows=rows)
            g1.record()
            torch.cuda.synchronize()
            ms = g0.elapsed_time(g1) / 5
        toks = B * S
        tf = GFLOP_PER_TOKEN * toks / (ms / 1e3) / 1e3
        print(f"B={B} S={S:5d} (M={B*S:5d}): {ms:8.1f}ms  "
              f"{toks/ms:7.0f} tok/s  ~{tf:5.1f} TFLOPS")
        del pg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
