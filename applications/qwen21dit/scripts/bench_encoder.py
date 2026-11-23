#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""encoder CPU 前向速度: MODE=eager|autocast|compile|compile-autocast (原生 compressed-tensors 加载)"""
from qwen21dit.paths import MODEL_DIR
import sys, time, torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

MODE = sys.argv[1] if len(sys.argv) > 1 else "eager"
ENC = str(MODEL_DIR / "fp8/qwen3vl8b_fp8")
TPL = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(ENC)
enc = Qwen3VLForConditionalGeneration.from_pretrained(ENC, torch_dtype=torch.bfloat16).eval()
lm = enc.model.language_model
hook = lm.norm.register_forward_hook(lambda m, a, o: a[0])  # pre-norm hidden (和 gen_one 一致)
print(f"[{time.time()-t0:.0f}s] 模型就绪", flush=True)

SHORT = "a lovely cat holding a wooden sign that says 'qwen fp8'"
LONG = ("A cinematic photograph of a young woman with long auburn hair standing in a neon-lit rainy street at night, "
        "wearing a translucent white shirt, holding a paper umbrella, reflections on wet asphalt, bokeh city lights, "
        "shot on 85mm lens, shallow depth of field, volumetric lighting, film grain, high detail skin texture. ") * 3

@torch.no_grad()
def once(prompt, ac):
    ids = tok(TPL.format(prompt), return_tensors="pt")["input_ids"]
    t = time.time()
    if ac:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = enc(ids, output_hidden_states=True, use_cache=False)
    else:
        out = enc(ids, output_hidden_states=True, use_cache=False)
    return time.time() - t, out.hidden_states[-1][0].float()

ac = MODE in ("autocast", "compile-autocast")
if MODE.startswith("compile"):
    t1 = time.time()
    enc = torch.compile(enc)
    d, _ = once(SHORT, ac)  # 触发编译
    print(f"[{time.time()-t1:.0f}s] compile 完成 (首次前向 {d*1000:.0f}ms)", flush=True)

for name, p in [("short", SHORT), ("long", LONG)]:
    nt = len(tok(TPL.format(p), return_tensors="pt")["input_ids"][0])
    once(p, ac)  # warm
    ds = [once(p, ac)[0] for _ in range(3)]
    print(f"{MODE:16s} {name:5s} {nt:4d} tok  " + "  ".join(f"{d*1000:7.0f}ms" for d in ds), flush=True)
