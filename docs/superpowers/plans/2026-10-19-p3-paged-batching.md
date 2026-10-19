# P3 Paged KV + Continuous Batching 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 paged KV cache 管理器、Triton paged decode attention kernel 与 continuous batching 调度循环；验收 = 并发输出与串行逐请求一致（fp32）+ 吞吐相对串行有可测提升。

**Architecture:** 分页存储 `[num_blocks, n_kv, block_size, head_dim]` + block_table 两级映射（vLLM B_Loc 模式，见 P2 阅读笔记）。prefill 走"临时 contiguous attention + scatter 写入分页"（每请求独立），decode 走 Triton kernel 直接按 block_table 寻址（每 program 处理一个 (请求, 头)，单 query 向量点积 + online softmax）。调度器做 iteration-level 装载/逐出。P1/P2 的单请求代码路径全部不动，P3 全部增量添加。

**Tech Stack:** 同前（triton 3.8.0 / gfx1201 / 纯脚本测试）。

**Spec:** `docs/superpowers/specs/2026-10-19-torchinfra-design.md` §3 P3

## Global Constraints

- python 一律 `~/rocm10/bin/python`，仓库根 `PYTHONPATH=.`
- P1/P2 既有测试必须始终通过（增量不动旧路径）
- 一致性验收用 fp32 双侧（沿用 P1/P2 手法）；bf16 只用于 demo/吞吐
- 默认参数：block_size=16，token 预算 8192（512 块），max_seqs=8

## 文件结构

```
torchinfra/
  paged_cache.py   # PagedKVCache：块池 + block_table + alloc/free/append/gather
  paged_attn.py    # Triton decode attention kernel + CacheView
  batch_model.py   # rope 批量 kernel + forward_decode 组装
  scheduler.py     # Request/Scheduler：continuous batching
tests/
  test_paged_cache.py test_paged_attn.py test_batch_model.py
  test_p3_scheduler.py test_p3_throughput.py
```

---

### Task 1: PagedKVCache 管理器

**Files:** Create `torchinfra/paged_cache.py`；Test `tests/test_paged_cache.py`

**Interfaces:**
- `PagedKVCache(n_layers, n_kv_heads, head_dim, num_blocks, block_size, dtype, device)`
  - 属性：`k/v: [n_layers, num_blocks, n_kv, block_size, head_dim]`；`block_table: [max_seqs, max_blocks_per_seq] int32`（-1 填充）；`kv_lens: [max_seqs] int32`
  - `alloc(req_id: int, n_tokens: int) -> int`（分配 `ceil(n_tokens/block_size)` 块写入 table，返回逻辑块数）
  - `extend(req_id: int, n_new_tokens: int)`（尾部追加块）
  - `free(req_id: int)`（块回收进 free list，table 清 -1）
  - `append(req_id: int, layer_idx: int, k, v, start: int)`（k/v `[1, n_kv, s, d]`，跨块拆写）
  - `gather(req_id: int, layer_idx: int, upto: int) -> (k, v)`（`[1, n_kv, upto, d]`，用于 prefill 参考与对拍）
  - `view()` 返回传给 kernel 的 `CacheView`（T2 定义）

- [ ] 失败测试：随机序列 append（含跨块 start），gather == 拼接参考；free 后块可复用且不污染
- [ ] 实现（free list 用 python `list`，块选择 LIFO；table 查找用 `.tolist()` 索引，教学版不追求分配性能）
- [ ] 通过 + commit `P3: paged KV cache 管理器`

核心写入逻辑（跨块拆分）：

```python
def append(self, req_id, layer_idx, k, v, start):
    s = k.shape[2]
    for t in range(s):  # 教学版逐 token；优化版按段拷贝留给进阶
        pos = start + t
        blk, off = self._locate(req_id, pos)
        self.k[layer_idx, blk, :, off] = k[0, :, t]
        self.v[layer_idx, blk, :, off] = v[0, :, t]
```

（`_locate` = table 查逻辑块 → 物理块；`gather` 用 `index_select` 按块取再 reshape 裁剪。）

### Task 2: prefill 路径（scatter 写入 + gather 对拍）

**Files:** Modify `torchinfra/paged_cache.py`（若无独立代码则本任务测试先行）；Test `tests/test_paged_attn.py` 的 prefill 部分

**Interfaces:**
- `prefill_request(model, req_id, ids: list[int], cache: PagedKVCache) -> logits [1, vocab]`：
  临时 `KVCache(n_layers=cfg.num_hidden_layers, max_seq=len(ids))`（P1 contiguous），
  跑 `model(ids, kv, 0, logits_to_keep="last")` 的**同一前向**需要在 attention 里
  同时把 k/v 镜像写入 paged——实现方式：临时 KVCache 子类 `MirroredKVCache`，
  `append` 时先调 `super().append` 再调 `cache.append(req_id, layer, k, v, start)`。
  返回末位 logits。
- [ ] 失败测试：prefill 三个不同长度请求后 `gather` 出的 K/V 与临时 contiguous cache 内容一致（fp32 逐位）
- [ ] 实现 `MirroredKVCache`（放 `torchinfra/paged_cache.py`）
- [ ] 通过 + commit `P3: prefill 镜像写入分页`

### Task 3: 批量 rope（per-request starts）

**Files:** Create `torchinfra/batch_model.py`；Test `tests/test_batch_model.py`

**Interfaces:**
- `_rope_decode_batch_kernel` + `rope_decode_batch(q, k, starts: Tensor[int])`：
  q/k `[B, heads, 1, head_dim]`（假连续陷阱同 P2，先 `_compact`），行主序
  `b = row // n_heads`，`pos = starts[b]`，half-split 原地旋转（复用 P2 kernel 的数学）
- [ ] 失败测试：B=4、starts=[7,0,33,1]，与逐请求 `TritonRoPE.apply_qk(x_b, start_b)` 对拍（fp32 1e-3）
- [ ] 实现（kernel 里 cos/sin 表从模块传入）
- [ ] 通过 + commit `P3: 批量 rope decode kernel`

### Task 4: Triton paged decode attention

**Files:** Create `torchinfra/paged_attn.py`；Test `tests/test_paged_attn.py`（追加）

**Interfaces:**
- `CacheView`（dataclass）：`k, v, block_table[int32, B, max_blocks], kv_lens[int32, B], num_blocks, block_size`
- `_paged_decode_kernel`：grid `(B, n_heads)`；每 program 单 (请求, 头)；逐块 load K `[BLOCK, D]`，向量点积 `qk = sum(k * q, 1)`，online softmax（`where(m==-inf)` 防 NaN，笔记技巧），`acc += sum(v * p, 1)`；块号 `tl.load(block_table + ...)` 后 `.to(tl.int64)`
- `paged_decode_attention(q [B, n_heads, 1, D], view) -> out [B, n_heads, D]`
- [ ] 失败测试：B=3 混合 kv_len（含非整块尾），fp32 下与"gather→P1 显式 attention 公式"参考对拍 1e-3；GQA 头映射覆盖（16Q/8KV）
- [ ] 实现（`q.to(tl.float32)` 全程 fp32 计算，存回 Out dtype）
- [ ] 通过 + commit `P3: Triton paged decode attention`

### Task 5: forward_decode 批量组装

**Files:** Modify `torchinfra/batch_model.py`、`torchinfra/model.py`（新增方法，不改旧签名）

**Interfaces:**
- `Qwen3ForCausalLM.forward_decode(token_ids [B,1], view: CacheView, starts: Tensor[int]) -> logits [B, vocab]`：
  复用 `self.model` 的 embed/层/norm/lm_head；attention 用 `forward_paged(hidden, rope_batch, view, layer_idx)`（在 `Attention` 上新增方法：q/k/v proj + q/k norm + `rope_decode_batch` + **每请求 append 新 k/v 到分页**（python 循环 B≤8）+ `paged_decode_attention`）
- [ ] 失败测试：B=3 各自 prefill 后，各 decode 一步的 logits == P1 引擎逐请求 decode 一步的 logits（fp32，`torch.testing.assert_close` rtol/atol 1e-3）
- [ ] 实现（append 在 attention 内做，保证 kernel 读到含新 token 的完整 kv_len）
- [ ] 通过 + commit `P3: forward_decode 批量一步`

### Task 6: Scheduler + 终局验收（并发==串行）

**Files:** Create `torchinfra/scheduler.py`；Test `tests/test_p3_scheduler.py`

**Interfaces:**
- `Request(rid, prompt_ids, max_new_tokens, temperature, top_p, seed)`（state: waiting/prefilled/finished；`output_ids`）
- `Scheduler(model, cache, max_seqs=8)`：`submit(prompt_ids, **kw) -> rid`；`step() -> list[完成事件]`；
  每步：① waiting→装载（受 max_seqs 与块预算）② 新请求逐个 prefill（采样首 token）③ 本轮**无新 prefill 时**对全部 running 做 decode 一步（vLLM prefill 优先语义）④ 逐请求采样/EOS/长度终止，`free` 块
- [ ] 失败测试：4 个不同 prompt、greedy、各 24 token——输出 ids 与 **P1 引擎逐请求**（fp32，同 backend）完全一致；含 EOS 提前终止用例
- [ ] 实现 + 通过 + commit `P3: continuous batching 调度器（并发==串行）`

### Task 7: 吞吐对比 + README

**Files:** Test `tests/test_p3_throughput.py`（输出数据不 assert 硬阈值，仅打印）；README 更新

- [ ] bf16：8 个 prompt（长中短混合）、max_new=64、greedy：串行（P1 generate 循环）vs 并发（Scheduler），打印各自 wall time 与总 tok/s，记录进 README
- [ ] 全量回归（P0/P1/P2/P3 全部测试）+ commit `P3: 吞吐对比`

---

## Self-Review 记录

- Spec 覆盖：块池/block table/分配释放复用（T1）✓；Triton paged attention decode 路径（T4）✓；continuous batching 混跑（T6：prefill 装载 + decode 轮转，prefill 优先）✓；并发==串行（T6 fp32）✓；吞吐提升（T7）✓。prefill 的 Triton kernel 明确不在本阶段（spec：prefill 后续）
- 占位符：T1 的 append 给出逐 token 教学实现并注明优化方向；其余代码在执行时按接口完整实现（接口签名已定）
- 类型一致性：`CacheView` T4 定义 T5 消费；`MirroredKVCache`/`forward_decode`/`Scheduler.step` 命名在任务间一致；`starts` 语义 = 各请求当前 kv_len（decode 前）
