"""Continuous batching 调度器：iteration-level 装载/逐出，prefill 优先。

decode 侧 CPU 开销优化（对照现代引擎）：
- 请求长度纯 python 记账（Request.kv_len），不读 GPU（无 tolist 同步）
- 采样走 sample_batch 整批 GPU（单次同步）
- 图按档位（1/2/4/8）惰性构建：单流不再陪跑 7/8 的 dummy
"""
from dataclasses import dataclass, field
import torch

from torchinfra.paged_cache import PagedKVCache, MirroredKVCache
from torchinfra.batch_model import decode_step
from torchinfra.sampling import sample_token, sample_batch


@dataclass
class Request:
    rid: int
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 0.0
    seed: int | None = None      # 仅 eager 路径生效；批采样用调度器级 generator
    output_ids: list[int] = field(default_factory=list)
    state: str = "waiting"       # waiting / running / finished
    kv_len: int = 0              # python 侧长度记账（prompt + 已入 cache 的 token）
    _gen: torch.Generator | None = None

    def __post_init__(self):
        if self.seed is not None:
            self._gen = torch.Generator(device="cuda")
            self._gen.manual_seed(self.seed)

    @property
    def done(self) -> bool:
        return (self.state == "finished"
                or len(self.output_ids) >= self.max_new_tokens)


class Scheduler:
    def __init__(self, model, tok, num_blocks: int = 384, block_size: int = 16,
                 max_seqs: int = 8, dtype=torch.bfloat16,
                 graph_mode: str = "piecewise",
                 buckets: tuple[int, ...] = (1, 2, 4, 8)):
        """graph_mode: piecewise（默认，最快）/ manual（整图捕获）/ none（eager）"""
        self.model = model
        self.tok = tok
        self.cache = PagedKVCache(model.cfg.num_hidden_layers,
                                  model.cfg.num_key_value_heads,
                                  model.cfg.head_dim, num_blocks=num_blocks,
                                  block_size=block_size, dtype=dtype)
        self.max_seqs = max_seqs
        self.graph_mode = graph_mode
        self.buckets = sorted(b for b in buckets if b <= max_seqs) or [max_seqs]
        self._runners: dict[int, object] = {}
        self._gen = torch.Generator(device="cuda")
        self._gen.manual_seed(0)
        self._waiting: list[Request] = []
        self._running: list[Request] = []
        self._next_rid = 0

    def submit(self, prompt_ids: list[int], **kw) -> int:
        req = Request(self._next_rid, list(prompt_ids), **kw)
        self._next_rid += 1
        self._waiting.append(req)
        return req.rid

    def _blocks_needed(self, req: Request) -> int:
        import math
        return math.ceil((len(req.prompt_ids) + req.max_new_tokens)
                         / self.cache.block_size)

    def _admit(self) -> list[Request]:
        """从 waiting 装载（受 max_seqs 与块预算约束），返回本轮要 prefill 的请求。"""
        admitted = []
        while self._waiting:
            req = self._waiting[0]
            if len(self._running) + len(admitted) >= self.max_seqs:
                break
            used = int((self.cache.block_table >= 0).sum())
            if used + self._blocks_needed(req) > self.cache.num_blocks:
                break
            self._waiting.pop(0)
            admitted.append(req)
        return admitted

    def _prefill(self, req: Request) -> None:
        # 全量预留（prompt+max_new）：GPU 侧 append kernel 无法扩表
        self.cache.reserve(req.rid, len(req.prompt_ids) + req.max_new_tokens)
        mirror = MirroredKVCache(self.cache, req_id=req.rid,
                                 max_seq=len(req.prompt_ids))
        with torch.no_grad():
            logits = self.model(
                torch.tensor([req.prompt_ids], device="cuda"), kv=mirror,
                start_pos=0, logits_to_keep="last")
        req.kv_len = len(req.prompt_ids)
        t = sample_token(logits[0, -1], req.temperature, 0, req.top_p, req._gen)
        if t in self.tok.eos_ids:
            req.state = "finished"
            self.cache.free(req.rid)
            return
        req.output_ids.append(t)
        req.state = "running"

    def _get_runner(self, n: int):
        size = next((b for b in self.buckets if b >= n), self.buckets[-1])
        runner = self._runners.get(size)
        if runner is None and self.graph_mode != "none":
            from torchinfra.fuse import fuse_decode_projections
            from torchinfra.cudagraph import (DecodeGraphRunner,
                                              DecodeGraphRunnerPiecewise)
            if not self._runners:        # 首次建图前做投影融合（幂等）
                fuse_decode_projections(self.model)
            cls = (DecodeGraphRunnerPiecewise if self.graph_mode == "piecewise"
                   else DecodeGraphRunner)
            runner = cls(self.model, self.cache, batch_size=size)
            self._runners[size] = runner
        return runner

    def step(self) -> list[Request]:
        """推进一个迭代：装载+prefill 新请求；无新 prefill 时 decode 全体一步。"""
        finished = []
        admitted = self._admit()
        if admitted:
            for req in admitted:          # prefill 优先：本轮不做 decode
                self._prefill(req)
                if req.state == "finished":
                    finished.append(req)
            self._running.extend(r for r in admitted if r.state == "running")
            self._running = [r for r in self._running if r.state != "finished"]
            return finished
        if not self._running:
            return finished
        reqs = self._running
        ids = [r.output_ids[-1] for r in reqs]
        runner = self._get_runner(len(reqs))
        if runner is not None:
            rows = [(r.rid, r.kv_len) for r in reqs]       # 纯 python，无同步
            with torch.no_grad():
                logits = runner.run(ids, rows)
            for r in reqs:
                r.kv_len += 1
            # kv_lens 批量回写（供 free/gather 一致性；一次 H2D）
            rids = torch.tensor([r.rid for r in reqs], device="cuda")
            lens = torch.tensor([r.kv_len for r in reqs], dtype=torch.int32,
                                device="cuda")
            self.cache.kv_lens[rids] = lens
            toks = sample_batch(logits,
                                temperature=[r.temperature for r in reqs],
                                top_p=[r.top_p for r in reqs],
                                generator=self._gen)
        else:
            with torch.no_grad():
                logits = decode_step(self.model,
                                     torch.tensor(ids, device="cuda").unsqueeze(1),
                                     self.cache, [r.rid for r in reqs])
            toks = [sample_token(row, r.temperature, 0, r.top_p, r._gen)
                    for req, row in zip(reqs, logits)
                    for r in [req]]
        for req, t in zip(reqs, toks):
            if t in self.tok.eos_ids:
                req.state = "finished"
                finished.append(req)
                self.cache.free(req.rid)
            else:
                req.output_ids.append(t)
                if req.done:
                    req.state = "finished"
                    finished.append(req)
                    self.cache.free(req.rid)
        self._running = [r for r in reqs if r.state != "finished"]
        return finished

    def warmup(self) -> None:
        """预热全部档位的图（触发投影融合与编译/捕获）。

        真实服务应在启动时调用；否则首次冷启动含数秒的 Inductor 编译税。
        预热运行写入 dummy 块，不影响业务数据。
        """
        if self.graph_mode == "none":
            return
        from torchinfra.fuse import fuse_decode_projections
        fuse_decode_projections(self.model)
        for b in self.buckets:
            self._get_runner(b)

    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    def run(self) -> dict[int, list[int]]:
        """跑完全部请求，返回 rid → 输出 token 列表。"""
        results: dict[int, list[int]] = {}
        while self.has_work():
            for req in self.step():
                results[req.rid] = req.output_ids
        return results
