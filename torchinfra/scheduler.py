"""Continuous batching 调度器：iteration-level 装载/逐出，prefill 优先。

decode 侧 CPU 开销优化（对照现代引擎，逐层递进）：
- 请求长度纯 python 记账（Request.kv_len），不读 GPU（无 tolist 同步）
- 图按档位（1/2/4/8...）惰性构建：单流不再陪跑 dummy
- 零同步稳态循环（P6，vLLM v1 overlap scheduling 的核心）：
  稳态每步仅 advance（lens+1 一次 kernel）→ replay → GPU 采样 →
  tokens 直接回填 → pinned 异步拷贝 + event（lag-1 完成检测）。
  热循环无任何主机同步，CPU 时间被 GPU 执行完全遮蔽；
  完成判定滞后一步（多算一步、丢弃），每请求生命周期仅一次。
- staging 仅在 batch 组合变化时发生：bt 行在 prefill reserve 后不变，
  lens 每步只是 +1——不再每步 zero_ + 逐行重拷 + 逐元素标量写。
"""
from dataclasses import dataclass, field
import torch

from torchinfra.paged_cache import PagedKVCache
from torchinfra.batch_model import decode_step
from torchinfra.sampling import sample_token, sample_batch, sample_batch_gpu


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

    def __post_init__(self):
        if self.seed is not None:
            self._gen = torch.Generator(device="cuda")
            self._gen.manual_seed(self.seed)
        else:
            self._gen = None

    @property
    def done(self) -> bool:
        return (self.state == "finished"
                or len(self.output_ids) >= self.max_new_tokens)


class Scheduler:
    def __init__(self, model, tok, num_blocks: int = 384, block_size: int = 16,
                 max_seqs: int = 8, dtype=torch.bfloat16,
                 graph_mode: str = "piecewise",
                 buckets: tuple[int, ...] = (1, 2, 4, 8)):
        """graph_mode: piecewise（默认，最快）/ manual（整图捕获）/ none（eager）
        buckets 必须覆盖 max_seqs（如 max_seqs=32 用 (1,2,4,8,16,32)）。"""
        self.model = model
        self.tok = tok
        self.cache = PagedKVCache(model.cfg.num_hidden_layers,
                                  model.cfg.num_key_value_heads,
                                  model.cfg.head_dim, num_blocks=num_blocks,
                                  block_size=block_size, dtype=dtype)
        self.max_seqs = max_seqs
        self.graph_mode = graph_mode
        self.buckets = sorted(b for b in buckets if b <= max_seqs) or [max_seqs]
        assert self.buckets[-1] >= max_seqs, \
            f"buckets {buckets} 覆盖不到 max_seqs={max_seqs}"
        self._runners: dict[int, object] = {}
        self._gen = torch.Generator(device="cuda")
        self._gen.manual_seed(0)
        self._waiting: list[Request] = []
        self._running: list[Request] = []
        self._next_rid = 0
        self._steady: dict | None = None   # 零同步稳态（组合不变期复用）

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

    def _prefill_batch(self, reqs: list[Request]) -> list[Request]:
        """批量 prefill：整轮 prompt 打包一次前向（权重只扫一遍）。

        P6 剖析：逐请求 prefill 在 B=32 占墙钟 2/3；批化后 GEMM 全在
        [sum_S, E] 上跑，注意力/ cache 写入走 varlen kernel。
        """
        from torchinfra.prefill import prefill_batch
        for req in reqs:
            self.cache.reserve(req.rid,
                               len(req.prompt_ids) + req.max_new_tokens)
        with torch.no_grad():
            logits = prefill_batch(self.model, self.cache,
                                   [r.prompt_ids for r in reqs],
                                   bt_rows=[r.rid for r in reqs])
        toks = sample_batch(logits,
                            temperature=[r.temperature for r in reqs],
                            top_p=[r.top_p for r in reqs],
                            generator=self._gen)
        finished = []
        for req, t in zip(reqs, toks):
            req.kv_len = len(req.prompt_ids)
            if t in self.tok.eos_ids:
                req.state = "finished"
                finished.append(req)
                self.cache.free(req.rid)
            else:
                req.output_ids.append(t)
                req.state = "running"
        return finished

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

    # ---- 零同步 decode 稳态（图路径）-------------------------------------

    def _try_consume(self, force_par: int | None = None) -> list[Request]:
        """消费已就绪的 pending 采样结果（滞后完成检测）。

        双缓冲流水线深度 2：query() 未就绪就缓一步（CPU 得以领先 GPU 两个
        decode 步，enqueue 时间被完全遮蔽）；仅在复写同一缓冲前
        （force_par）或组合收尾时阻塞同步。消费顺序 p0→p1 保持 token 序。
        """
        finished: list[Request] = []
        st = self._steady
        if st is None:
            return finished
        for p in ([force_par] if force_par is not None else (0, 1)):
            pend = st["pend"][p]
            if pend is None:
                continue
            if force_par is None and not pend["ev"].query():
                continue                      # 在途：缓一步，不阻塞 CPU
            pend["ev"].synchronize()
            ids = pend["pin"][:pend["n"]].tolist()   # pinned → 纯主机读
            for r, t in zip(pend["reqs"], ids):
                if r.state != "running":
                    continue
                if t in self.tok.eos_ids:
                    r.state = "finished"
                    finished.append(r)
                    self.cache.free(r.rid)
                else:
                    r.output_ids.append(t)
                    if r.done:
                        r.state = "finished"
                        finished.append(r)
                        self.cache.free(r.rid)
            st["pend"][p] = None
        return finished

    def _start_steady(self) -> None:
        reqs = list(self._running)
        n = len(reqs)
        runner = self._get_runner(n)
        assert runner is not None
        runner.stage([r.output_ids[-1] for r in reqs],
                     [(r.rid, r.kv_len) for r in reqs])
        # 采样器与参数张量组合期一次性构建（稳态零分配、零同步）
        dev = "cuda"
        temps = torch.tensor([r.temperature for r in reqs],
                             dtype=torch.float32, device=dev).view(n, 1)
        tps = torch.tensor([r.top_p for r in reqs],
                           dtype=torch.float32, device=dev).view(n, 1)
        all_greedy = all(r.temperature <= 0 for r in reqs)
        any_topp = any(0.0 < r.top_p < 1.0 for r in reqs)
        gen = self._gen
        if all_greedy:
            def sampler(logits):
                return logits.argmax(-1)
        else:
            def sampler(logits):
                return sample_batch_gpu(logits, temps, tps, any_topp, gen)
        self._steady = {
            "sig": tuple(r.rid for r in reqs),
            "reqs": reqs, "n": n, "runner": runner, "sampler": sampler,
            "pin": (torch.zeros(runner.B, dtype=torch.long, pin_memory=True),
                    torch.zeros(runner.B, dtype=torch.long, pin_memory=True)),
            "ev": (torch.cuda.Event(), torch.cuda.Event()),
            "pend": [None, None],
            "par": 0,
        }
        return self._steady_tail()

    def _steady_tail(self) -> list[Request]:
        """一次稳态 decode：replay → GPU 采样 → 回填 → pinned 异步记录。

        复写缓冲前先强制消费它两步前的 pending（此时 CPU 领先 ≥2 步，
        阻塞等待等价于限深 2——GPU 是瓶颈时 CPU 仍在遮蔽窗口内）。
        返回本次强制消费判定的 finished（含 token，供 run() 收集结果）。
        """
        st = self._steady
        run, n = st["runner"], st["n"]
        p = st["par"]
        finished = self._try_consume(force_par=p)
        with torch.no_grad():
            logits = run.replay()[:n]
            ids = st["sampler"](logits)
            run.feed(ids)
        st["pin"][p][:n].copy_(ids, non_blocking=True)
        st["ev"][p].record()
        st["pend"][p] = {"pin": st["pin"][p], "ev": st["ev"][p],
                         "reqs": st["reqs"], "n": n}
        st["par"] = p ^ 1
        for r in st["reqs"]:
            r.kv_len += 1
        return finished

    def _decode_async(self) -> list[Request]:
        finished = self._try_consume()
        if any(r.state == "finished" for r in self._running):
            # 组合收尾：排干双缓冲在途 pending（阻塞；组合变化是罕见事件）
            finished += self._try_consume(force_par=0)
            finished += self._try_consume(force_par=1)
            self._running = [r for r in self._running
                             if r.state != "finished"]
            self._steady = None
        if not self._running:
            return finished
        sig = tuple(r.rid for r in self._running)
        if self._steady is None or self._steady["sig"] != sig:
            return finished + self._start_steady()   # 组合变化：staging + 首步
        self._steady["runner"].advance()
        return finished + self._steady_tail()

    # ---- 迭代入口 ---------------------------------------------------------

    def step(self) -> list[Request]:
        """推进一个迭代：装载+prefill 新请求；无新 prefill 时 decode 全体一步。"""
        admitted = self._admit()
        if admitted:
            finished = self._prefill_batch(admitted)   # 整轮打包 prefill
            self._running.extend(r for r in admitted if r.state == "running")
            self._running = [r for r in self._running if r.state != "finished"]
            return finished
        if not self._running:
            return []
        if self.graph_mode != "none":
            return self._decode_async()

        # eager 回退路径（无图）
        reqs = self._running
        finished = []
        with torch.no_grad():
            logits = decode_step(
                self.model,
                torch.tensor([r.output_ids[-1] for r in reqs],
                             device="cuda").unsqueeze(1),
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
