# tests/test_p3_throughput.py
"""P3 吞吐对比：串行（P1 路径逐请求）vs continuous batching（bf16）。"""
import time
import torch
from tests._ref import MODEL_DIR
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.kv_cache import KVCache
from torchinfra.scheduler import Scheduler
from torchinfra.sampling import sample_token
from torchinfra.tokenizer import TokenizerBox

PROMPTS = [
    "请用三句话介绍一下长城的历史。",
    "Explain how a bicycle works.",
    "写一首关于春天的短诗。",
    "What are the benefits of exercise?",
    "简要说明光合作用的过程。",
    "How do computers store data?",
    "用中文解释什么是分页内存管理。",
    "Describe the water cycle briefly.",
]
N_NEW = 256


def serial(model, tok) -> tuple[float, int]:
    total_tokens = 0
    t0 = time.perf_counter()
    for p in PROMPTS:
        ids = tok.encode(p)
        kv = KVCache(model.cfg.num_hidden_layers, len(ids) + N_NEW + 8,
                     model.cfg.num_key_value_heads, model.cfg.head_dim)
        with torch.no_grad():
            logits = model(torch.tensor([ids], device="cuda"), kv=kv,
                           start_pos=0, logits_to_keep="last")
            pos = len(ids)
            for _ in range(N_NEW):
                t = sample_token(logits[0, -1])
                total_tokens += 1
                logits = model(torch.tensor([[t]], device="cuda"), kv=kv,
                               start_pos=pos, logits_to_keep="last")
                pos += 1
    dt = time.perf_counter() - t0
    return dt, total_tokens


def batched(model, tok) -> tuple[float, int]:
    sched = Scheduler(model, tok)
    for p in PROMPTS:
        sched.submit(tok.encode(p), max_new_tokens=N_NEW)
    sched.warmup()                      # 图编译/捕获在计时外（服务启动时预热）
    # prefill 路径的 Triton kernel/SDPA 首调初始化也移出计时窗
    from torchinfra.paged_cache import PagedKVCache
    from torchinfra.prefill import prefill_batch
    p0 = PagedKVCache(model.cfg.num_hidden_layers,
                      model.cfg.num_key_value_heads, model.cfg.head_dim,
                      num_blocks=8, block_size=16)
    p0.reserve(0, 8)
    prefill_batch(model, p0, [[1, 5, 9, 3, 7, 2, 8, 4]], bt_rows=[0])
    t0 = time.perf_counter()
    results = sched.run()
    dt = time.perf_counter() - t0
    return dt, sum(len(v) for v in results.values())


def main():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR)  # bf16
    tok = TokenizerBox(MODEL_DIR)

    dt_s, n_s = serial(model, tok)      # 先串行（同时当预热）
    dt_b, n_b = batched(model, tok)
    print(f"串行: {n_s} tokens / {dt_s:.1f}s = {n_s / dt_s:.1f} tok/s")
    print(f"并发: {n_b} tokens / {dt_b:.1f}s = {n_b / dt_b:.1f} tok/s")
    print(f"吞吐提升: {(n_s / dt_s) / (n_b / dt_b) and (n_b / dt_b) / (n_s / dt_s):.2f}x")


if __name__ == "__main__":
    main()
