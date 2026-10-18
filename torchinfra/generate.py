"""prefill + decode 两阶段流式生成。"""
from typing import Iterator
import torch
from torchinfra.kv_cache import KVCache
from torchinfra.sampling import sample_token


@torch.no_grad()
def generate(model, tok, prompt_ids: list[int], max_new_tokens: int = 128,
             temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
             seed: int | None = None) -> Iterator[tuple[int, str]]:
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
    dtype = next(model.parameters()).dtype
    kv = KVCache(model.cfg.num_hidden_layers, model.max_seq,
                 model.cfg.num_key_value_heads, model.cfg.head_dim,
                 dtype=dtype)
    ids = list(prompt_ids)
    logits = model(torch.tensor([ids], device="cuda"), kv=kv, start_pos=0,
                   logits_to_keep="last")
    pos = len(ids)
    text = ""
    for _ in range(max_new_tokens):
        t = sample_token(logits[0, -1], temperature, top_k, top_p, gen)
        if t in tok.eos_ids:
            break
        ids.append(t)
        new_text = tok.decode(ids)      # 简单起见整段重解码，O(n²) 学习项目可接受
        delta, text = new_text[len(text):], new_text
        yield t, delta
        if pos + 1 >= model.max_seq:
            break
        logits = model(torch.tensor([[t]], device="cuda"), kv=kv,
                       start_pos=pos, logits_to_keep="last")
        pos += 1
