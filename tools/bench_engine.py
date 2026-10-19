"""端到端引擎基准：batch 扫描（权重扫描 1.19GB/步 是固定成本，B 越大摊越薄）。

用法：PYTHONPATH=. python tools/bench_engine.py [b1 b2 ...]
默认扫 1 8 16 32；llama.cpp 参照（同卡同模型 bf16 单流 213.4 tok/s）。
"""
import sys
import time
import torch

from tests._ref import MODEL_DIR
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.scheduler import Scheduler
from torchinfra.tokenizer import TokenizerBox

PROMPT = "请写一段关于秋天的文字，并解释其中的意象。"
N_NEW = 256


def bench(model, tok, batch: int) -> float:
    buckets = [b for b in (1, 2, 4, 8, 16, 32) if b <= batch]
    if buckets[-1] < batch:
        buckets.append(batch)
    sched = Scheduler(model, tok, num_blocks=max(384, batch * 24),
                      max_seqs=batch, buckets=tuple(buckets))
    sched.warmup()
    for _ in range(batch):
        sched.submit(tok.encode(PROMPT), max_new_tokens=N_NEW)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = sched.run()
    dt = time.perf_counter() - t0
    n = sum(len(v) for v in results.values())
    del sched
    torch.cuda.empty_cache()
    return n / dt


def main():
    torch.manual_seed(0)
    batches = [int(a) for a in sys.argv[1:]] or [1, 8, 16, 32]
    # 注意：compile=True 的 prefill 与 decode 的 piecewise cudagraph trees
    # 冲突（prefill 插在两次回放之间会让 Inductor 跳过 cudagraph，
    # decode 退化 ~2.3x）——prefill 保持 eager，仅预热首次 forward 的
    # 冷启动（kernel 缓存/分配器），移出计时窗。
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR)   # bf16
    tok = TokenizerBox(MODEL_DIR)
    from torchinfra.kv_cache import KVCache
    kv = KVCache(model.cfg.num_hidden_layers, 16,
                 model.cfg.num_key_value_heads, model.cfg.head_dim)
    with torch.no_grad():
        model(torch.zeros(1, 4, dtype=torch.long, device="cuda"), kv=kv,
              start_pos=0, logits_to_keep="last")
    print(f"Qwen3-0.6B bf16 @ 9070 GRE，每请求 {N_NEW} new tokens\n")
    for b in batches:
        tps = bench(model, tok, b)
        line = f"B={b:2d}  {tps:7.1f} tok/s"
        if b == 1:
            line += f"（llama.cpp 参照 213.4，{tps / 213.4 * 100:.0f}%）"
        print(line)


if __name__ == "__main__":
    main()
