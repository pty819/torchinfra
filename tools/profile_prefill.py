"""prefill 剖析：GEMM / pointwise / attention / cache 写入的耗时分布。

prefill 是计算瓶颈（区别于 decode 的带宽瓶颈）——检查离卡的理论
TFLOPS 还有多远。用法：PYTHONPATH=. python tools/profile_prefill.py [B]
"""
import sys
import time
import torch

from tests._ref import MODEL_DIR
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.paged_cache import PagedKVCache
from torchinfra.prefill import prefill_batch
from torchinfra.tokenizer import TokenizerBox

PROMPT = "请写一段关于秋天的文字，并解释其中的意象。"


def bench(model, paged, prompts, iters=20):
    with torch.no_grad():
        for _ in range(3):
            prefill_batch(model, paged, prompts, bt_rows=list(range(len(prompts))))
        torch.cuda.synchronize()
        g0 = torch.cuda.Event(enable_timing=True)
        g1 = torch.cuda.Event(enable_timing=True)
        g0.record()
        for _ in range(iters):
            prefill_batch(model, paged, prompts,
                          bt_rows=list(range(len(prompts))))
        g1.record()
        torch.cuda.synchronize()
        return g0.elapsed_time(g1) / iters


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    torch.manual_seed(0)
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR)   # bf16
    tok = TokenizerBox(MODEL_DIR)
    ids = tok.encode(PROMPT)
    S = len(ids)
    paged = PagedKVCache(model.cfg.num_hidden_layers,
                         model.cfg.num_key_value_heads, model.cfg.head_dim,
                         num_blocks=B * (S + 256) // 16 + 8, block_size=16)
    for r in range(B):
        paged.reserve(r, S)
    prompts = [ids] * B

    ms = bench(model, paged, prompts)
    total_tokens = B * S
    flops = 2 * 660e6 * total_tokens        # 前向 ~2×参数量/token（粗）
    print(f"B={B}: {ms:.1f}ms / {total_tokens} tokens = "
          f"{total_tokens / ms:.0f} tok/s，~{flops / (ms / 1e3) / 1e12:.1f} TFLOPS")

    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                prefill_batch(model, paged, prompts,
                              bt_rows=list(range(B)))
            torch.cuda.synchronize()
    tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=14)
    import re
    for line in tbl.split("\n"):
        if re.search(r"Cijk|triton|_prefill|_append|elementwise|reduce|"
                     r"copy|vectorized|CatArray|index", line):
            print(line[:150])


if __name__ == "__main__":
    main()
