"""jina-reranker-v3.5 的并发服务：调度器把多个请求的块拼进同一次前向。

对外是 vLLM / Jina 那套 rerank 接口，不是 chat completion：

    POST /v1/rerank
    {"model": "...", "query": "...", "documents": ["..."], "top_n": 3}

GPU 上只有调度线程会跑前向。HTTP 线程把请求放进队列，等一小段时间把同时到达的
请求收成一批，再按 max_batch 条 prompt 左 padding 后一次算完。
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
        再把所有块按 max_batch 拼成若干次前向。"""
        reranker = self.reranker
        jobs = []
        for query, documents, top_n, fut in batch:
            if not documents:
                fut.set_result(([], 0))
                continue
            query_t, blocks = reranker.plan_blocks(query, documents)
            jobs.append({
                "fut": fut, "query": query_t, "blocks": blocks,
                "documents": documents, "top_n": top_n,
                "docs": [], "queries": [], "weights": [], "tokens": 0,
            })
        pending = [(j, block) for j in jobs for block in j["blocks"]]
        while pending:
            chunk = pending[:self.max_batch]
            pending = pending[self.max_batch:]
            prompts = [_format_prompt(job["query"], block, None, reranker.tok)
                       for job, block in chunk]
            encoded = reranker.tok(prompts, padding=True, return_tensors="pt")
            ids = encoded.input_ids.to("cuda")
            valid = encoded.attention_mask.bool().to("cuda")
            # CPU 侧判定是否有 padding：避免前向里 GPU→CPU 同步
            padded = not bool(encoded.attention_mask.all())
            rows = reranker.embed_batch(ids, valid, padded)
            self.forwards += 1
            for i, ((job, _), (doc, qry)) in enumerate(zip(chunk, rows)):
                scores = F.cosine_similarity(doc, qry.expand_as(doc), dim=-1)
                job["docs"].append(doc)
                job["queries"].append(qry)
                job["weights"].append(float(((1.0 + scores) / 2.0).max()))
                job["tokens"] += int(valid[i].sum())
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
