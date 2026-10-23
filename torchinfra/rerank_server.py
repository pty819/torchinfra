"""jina-reranker-v3.5 的并发服务：调度器把多个请求的块拼进同一次前向。

对外是 vLLM / Jina 那套 rerank 接口，不是 chat completion：

    POST /v1/rerank
    {"model": "...", "query": "...", "documents": ["..."], "top_n": 3}

GPU 上只有调度线程会跑前向。HTTP 线程把请求放进队列，等一小段时间把同时到达的
请求收成一批；块按真实 token 长度分组——小块相近长度左 padding 拼批，大块只与
等长块拼（保 flash 路径）否则单独前向（详见 RerankScheduler._run）。
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from torchinfra.jina_rerank import JinaReranker, _format_prompt


class RerankScheduler:
    def __init__(self, reranker: JinaReranker, max_batch: int = 8,
                 coalesce_ms: float = 8.0):
        self.reranker = reranker
        self.max_batch = max_batch
        self.coalesce_ms = coalesce_ms
        self.forwards = 0
        tok = reranker.tok
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def rerank(self, query: str, documents: list[str],
               top_n: int | None = None, timeout: float = 120.0) -> list[dict]:
        fut: Future = Future()
        self._q.put((query, documents, top_n, fut))
        return fut.result(timeout)

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            batch = [item]
            deadline = time.perf_counter() + self.coalesce_ms / 1000
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._q.put(None)
                    break
                batch.append(nxt)
            try:
                self._run(batch)
            except Exception as exc:
                for *_, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)

    @torch.no_grad()
    def _run(self, batch) -> None:
        """batch: [(query, documents, top_n, future)]。每条请求先按官方规则切块，
        逐块编码拿真实 token 长度，再按长度分批前向：

        - 小块（≤512 tok）与长度相近的块左 padding 拼批（浪费小，
          attn_mask 后端在这个尺寸也便宜）；
        - 大块只与**等长**块拼（无 padding → is_causal flash 路径），
          否则单独 B=1 前向（吃 CUDA graph 分桶）。

        异长左 padding 在大 S 下是双重毒药：小块被撑到大块长度白算十几倍，
        且 padded 批的 SDPA 走 attn_mask 后端、物化 B×H×S² 注意力矩阵
        （掉出 flash）——B=2、S≈2900 实测慢 7 倍，更大 S 直接 OOM。"""
        reranker = self.reranker
        jobs = []
        for query, documents, top_n, fut in batch:
            if not documents:
                fut.set_result(([], 0))
                continue
            query_t, blocks = reranker.plan_blocks(query, documents)
            nb = len(blocks)
            jobs.append({
                "fut": fut, "query": query_t, "blocks": blocks,
                "documents": documents, "top_n": top_n,
                "docs": [None] * nb, "queries": [None] * nb,
                "weights": [None] * nb, "tokens": 0,
            })
        # 两遍编码：先逐条拿真实 token 长度（fast tokenizer，ms 级）
        items = []
        for job in jobs:
            for bidx, block in enumerate(job["blocks"]):
                prompt = _format_prompt(job["query"], block, None, reranker.tok)
                n = len(reranker.tok(prompt)["input_ids"])
                items.append((n, prompt, job, bidx))
        items.sort(key=lambda x: x[0])
        runs, i = [], 0
        while i < len(items):
            j = i + 1
            if items[i][0] <= 512:
                while (j < len(items) and items[j][0] <= 512
                       and items[j][0] <= items[i][0] * 1.25 + 64
                       and j - i < self.max_batch):
                    j += 1
            else:  # 大块：只拼完全等长的（编码后无 padding）
                while (j < len(items) and items[j][0] == items[i][0]
                       and j - i < self.max_batch):
                    j += 1
            runs.append(items[i:j])
            i = j
        for run in runs:
            prompts = [p for _, p, _, _ in run]
            encoded = reranker.tok(prompts, padding=True, return_tensors="pt")
            ids = encoded.input_ids.to("cuda")
            am = encoded.attention_mask
            # 单块走 valid=None：embed_batch 对 B=1 启用 CUDA graph 分桶
            valid = am.bool().to("cuda") if len(run) > 1 else None
            padded = not bool(am.all())
            rows = reranker.embed_batch(ids, valid, padded)
            self.forwards += 1
            for k, ((_, _, job, bidx), (doc, qry)) in enumerate(zip(run, rows)):
                scores = F.cosine_similarity(doc, qry.expand_as(doc), dim=-1)
                job["docs"][bidx] = doc
                job["queries"][bidx] = qry
                job["weights"][bidx] = float(((1.0 + scores) / 2.0).max())
                job["tokens"] += (int(valid[k].sum()) if valid is not None
                                  else int(am[0].sum()))
        for job in jobs:
            if job["fut"].done():
                continue
            result = reranker.combine(
                job["docs"], job["queries"], job["weights"],
                job["documents"], job["top_n"], return_embeddings=False)
            for row in result:
                row.pop("embedding", None)
            job["fut"].set_result((result, job["tokens"]))


def _documents(raw) -> list[str]:
    out = []
    for doc in raw:
        if isinstance(doc, str):
            out.append(doc)
        elif isinstance(doc, dict) and "text" in doc:
            out.append(doc["text"])
        else:
            raise ValueError("documents 里必须是字符串或 {\"text\": ...}")
    return out


def serve(model_dir: str, host: str = "127.0.0.1", port: int = 8000,
          max_batch: int = 8, coalesce_ms: float = 8.0) -> None:
    reranker = JinaReranker.from_pretrained(model_dir)
    sched = RerankScheduler(reranker, max_batch=max_batch,
                            coalesce_ms=coalesce_ms)
    model_name = "jina-reranker-v3.5"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/v1/health"):
                self._send(200, {"status": "ok"})
            elif path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": model_name, "object": "model", "owned_by": "jinaai"}]})
            else:
                self._send(404, {"error": {"message": f"未知路径 {path}"}})

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path not in ("/v1/rerank", "/rerank"):
                self._send(404, {"error": {"message": f"未知路径 {path}"}})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                query = req["query"]
                documents = _documents(req["documents"])
                top_n = req.get("top_n")
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": {"message": str(exc)}})
                return
            try:
                results, tokens = sched.rerank(query, documents, top_n)
            except Exception as exc:
                self._send(500, {"error": {"message": str(exc)}})
                return
            show_doc = req.get("return_documents", True)
            self._send(200, {
                "id": "rerank-" + uuid.uuid4().hex[:12],
                "model": req.get("model", model_name),
                "results": [{
                    "index": r["index"],
                    "relevance_score": r["relevance_score"],
                    **({"document": {"text": r["document"]}} if show_doc else {}),
                } for r in results],
                "usage": {"total_tokens": tokens},
            })

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"rerank api http://{host}:{port}/v1/rerank  "
          f"max_batch={max_batch} coalesce={coalesce_ms}ms")
    httpd.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser(description="jina-reranker-v3.5 OpenAI 风格 rerank 服务")
    p.add_argument("--model", default="~/models/jina-reranker-v3.5-fp8")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--coalesce-ms", type=float, default=8.0)
    args = p.parse_args()
    serve(args.model, args.host, args.port, args.max_batch, args.coalesce_ms)


if __name__ == "__main__":
    main()
