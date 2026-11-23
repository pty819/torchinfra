import os
"""rerank 端到端延迟基准：分解 plan(tokenize) / encode / 前向 / 合并。

用法：
    PYTHONPATH=. ~/rocm10/bin/python tools/bench_rerank_e2e.py [--bf16] [--n N]
"""
import argparse
import time

import torch
import torch.nn.functional as F

from torchinfra.jina_rerank import JinaReranker, _format_prompt

FP8 = os.path.expanduser("~/models/jina-reranker-v3.5-fp8")
BF16 = os.path.expanduser("~/models/jina-reranker-v3.5")

WORDS = ("memory recall embedding vector search index latency kernel "
         "gradient attention transformer window causal mask quantization "
         "cache scheduler batch throughput benchmark random passage text "
         "green tea antioxidant catchin health benefit research study "
         "market price coffee seasonal harvest robusta arabica espresso").split()


def synth_doc(tok, n_tokens: int, seed: int) -> str:
    """生成 token 数 ≈ n_tokens 的文档（tokenize 后裁剪，decode 回文本）。"""
    r = torch.Generator().manual_seed(seed)
    words = [WORDS[int(torch.randint(0, len(WORDS), (1,), generator=r))]
             for _ in range(int(n_tokens * 1.8) + 8)]
    doc = " ".join(words)
    ids = tok(doc, add_special_tokens=False)["input_ids"]
    if len(ids) > n_tokens:
        doc = tok.decode(ids[:n_tokens])
    return doc


class Timer:
    def __init__(self):
        self.parts = {}

    def add(self, name, dt):
        self.parts[name] = self.parts.get(name, 0.0) + dt

    @staticmethod
    def now():
        torch.cuda.synchronize()
        return time.perf_counter()


def run_once(r: JinaReranker, query: str, docs: list[str], t: Timer):
    t0 = Timer.now()
    query_t, blocks = r.plan_blocks(query, docs)
    t.add("plan", Timer.now() - t0)
    doc_embeds, query_embeds, weights = [], [], []
    fwd_tokens = 0
    for block in blocks:
        t0 = Timer.now()
        prompt = _format_prompt(query_t, block, None, r.tok)
        t.add("format", Timer.now() - t0)
        t0 = Timer.now()
        ids = r.tok(prompt, return_tensors="pt").input_ids.to("cuda")
        t.add("encode", Timer.now() - t0)
        fwd_tokens += ids.shape[1]
        t0 = Timer.now()
        doc, qry = r._embed_ids(ids)
        t.add("forward", Timer.now() - t0)
        scores = F.cosine_similarity(doc, qry.expand_as(doc), dim=-1)
        doc_embeds.append(doc)
        query_embeds.append(qry)
        weights.append(float(((1.0 + scores) / 2.0).max()))
    t0 = Timer.now()
    result = r.combine(doc_embeds, query_embeds, weights, docs, None, False)
    t.add("combine", Timer.now() - t0)
    return fwd_tokens, len(blocks), result


def bench(r: JinaReranker, query: str, docs: list[str], n: int, label: str):
    run_once(r, query, docs, Timer())  # warmup
    total = []
    agg = {}
    for _ in range(n):
        t = Timer()
        t0 = Timer.now()
        run_once(r, query, docs, t)
        t.add("TOTAL", Timer.now() - t0)
        total.append(t.parts["TOTAL"])
        for k, v in t.parts.items():
            agg[k] = agg.get(k, 0.0) + v
    med = sorted(total)[len(total) // 2] * 1e3
    parts = {k: v / n * 1e3 for k, v in agg.items()}
    print(f"{label:24s} e2e {med:7.1f} ms | " +
          " ".join(f"{k} {v:6.1f}" for k, v in parts.items() if k != "TOTAL"))
    return med


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--only", default=None, help="只跑指定 label，如 100x256")
    args = p.parse_args()
    path = BF16 if args.bf16 else FP8
    r = JinaReranker.from_pretrained(path)
    query = "What are the health benefits of green tea?"
    shapes = [
        ("8x64", 8, 64),
        ("32x256", 32, 256),
        ("100x256", 100, 256),
        ("20x512", 20, 512),
        ("200x512", 200, 512),
    ]
    for label, ndoc, dtok in shapes:
        if args.only and label != args.only:
            continue
        docs = [synth_doc(r.tok, dtok, i) for i in range(ndoc)]
        n_blk = None
        t = Timer()
        toks, n_blk, _ = run_once(r, query, docs, t)
        print(f"--- {label}: {toks} tok, {n_blk} block ---")
        bench(r, query, docs, args.n, label)


if __name__ == "__main__":
    main()
