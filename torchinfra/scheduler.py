"""Continuous batching 调度器：iteration-level 装载/逐出，prefill 优先。"""
from dataclasses import dataclass, field
import torch

from torchinfra.paged_cache import PagedKVCache, MirroredKVCache
from torchinfra.batch_model import decode_step
from torchinfra.sampling import sample_token


@dataclass
class Request:
    rid: int
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 0.0
    seed: int | None = None
    output_ids: list[int] = field(default_factory=list)
    state: str = "waiting"          # waiting / running / finished
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
                 max_seqs: int = 8, dtype=torch.bfloat16):
        self.model = model
        self.tok = tok
        self.cache = PagedKVCache(model.cfg.num_hidden_layers,
                                  model.cfg.num_key_value_heads,
                                  model.cfg.head_dim, num_blocks=num_blocks,
                                  block_size=block_size, dtype=dtype)
        self.max_seqs = max_seqs
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
        mirror = MirroredKVCache(self.cache, req_id=req.rid,
                                 max_seq=len(req.prompt_ids))
        with torch.no_grad():
            logits = self.model(
                torch.tensor([req.prompt_ids], device="cuda"), kv=mirror,
                start_pos=0, logits_to_keep="last")
        t = sample_token(logits[0, -1], req.temperature, 0, req.top_p, req._gen)
        if t in self.tok.eos_ids:
            req.state = "finished"
            self.cache.free(req.rid)
            return
        req.output_ids.append(t)
        req.state = "running"

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
        with torch.no_grad():
            logits = decode_step(self.model,
                                 torch.tensor(ids, device="cuda").unsqueeze(1),
                                 self.cache, [r.rid for r in reqs])
        for req, row in zip(reqs, logits):
            t = sample_token(row, req.temperature, 0, req.top_p, req._gen)
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

    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    def run(self) -> dict[int, list[int]]:
        """跑完全部请求，返回 rid → 输出 token 列表。"""
        results: dict[int, list[int]] = {}
        while self.has_work():
            for req in self.step():
                results[req.rid] = req.output_ids
        return results
