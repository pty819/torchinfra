# tests/test_p3_scheduler.py
"""P3 终局验收：continuous batching 并发输出与 P1 引擎串行逐请求一致（fp32）。"""
import torch
from tests._ref import MODEL_DIR
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.kv_cache import KVCache
from torchinfra.scheduler import Scheduler
from torchinfra.sampling import sample_token
from torchinfra.tokenizer import TokenizerBox

PROMPTS = [
    "请用一句话介绍你自己。",
    "What is paged attention?",
    "写一个关于秋天的短句。",
    "1+1等于几？",
]
N_NEW = 24


def p1_serial_generate(model, ids: list[int], n_new: int) -> list[int]:
    """P1 引擎串行参考（contiguous KV cache 路径）。"""
    kv = KVCache(model.cfg.num_hidden_layers, len(ids) + n_new + 8,
                 model.cfg.num_key_value_heads, model.cfg.head_dim,
                 dtype=torch.float32)
    outs = []
    with torch.no_grad():
        logits = model(torch.tensor([ids], device="cuda"), kv=kv, start_pos=0,
                       logits_to_keep="last")
        pos = len(ids)
        for _ in range(n_new):
            t = sample_token(logits[0, -1])
            outs.append(t)
            logits = model(torch.tensor([[t]], device="cuda"), kv=kv,
                           start_pos=pos, logits_to_keep="last")
            pos += 1
    return outs


def main():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    tok = TokenizerBox(MODEL_DIR)
    refs = {}
    for p in PROMPTS:
        ids = tok.encode(p)
        refs[p] = p1_serial_generate(model, ids, N_NEW)

    sched = Scheduler(model, tok, dtype=torch.float32)
    for p in PROMPTS:
        sched.submit(tok.encode(p), max_new_tokens=N_NEW)
    results = sched.run()

    for p in PROMPTS:
        rid = PROMPTS.index(p)
        got = results[rid]
        exp = refs[p]
        assert got == exp, f"[{p}] 不一致:\n got={got}\n exp={exp}"
    print(f"acceptance OK: {len(PROMPTS)} 个请求并发 == 串行，各 {N_NEW} tokens")
    for p in PROMPTS[:2]:
        print(" ", p, "→", tok.decode(results[PROMPTS.index(p)])[:60])


if __name__ == "__main__":
    main()
