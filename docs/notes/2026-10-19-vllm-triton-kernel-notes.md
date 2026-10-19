# vLLM Triton Flash Attention（prefix_prefill）阅读笔记

- 日期：2026-10-19
- 对象：`vllm/v1/attention/ops/prefix_prefill.py` 的 `_fwd_kernel`（源自 LightLLM
  的 context_flashattention_nopad，vLLM 官方 Triton attention 路径的核心之一）
- 阅读动机：它就是 P3 要写的"分页 KV + Triton attention"的工业级对照物

## 1. grid / block 划分

`grid = (batch, head, cdiv(max_input_len, BLOCK_M))`——三维 program id 各自对应
一个 (请求, 注意力头, query 的 M 块)。注意两个工程细节：

- 用 **max**_input_len 而不是每个请求的真实长度算第三维，配合 kernel 开头读
  `B_Seqlen/B_Start_Loc` 自行判断实际边界，浪费的 program 直接 return。这换来了
  统一 grid（CUDA Graph 友好）。
- M 走 query 维、N 走 KV 维；BLOCK_M=128 / BLOCK_N=64 / BLOCK_SIZE(KV tile)=32，
  与物理页大小（PHYSICAL_BLOCK_SIZE，如 16/544）解耦——tile 跨页由
  `_paged_kv_cache_offsets` 逐 token 换算处理。

## 2. 分页 KV 的 block table 索引（P3 的直接蓝本）

每个 token 的物理地址两级映射：逻辑块号 `bn = B_Loc[batch, pos // PAGE]`，
块内偏移 `internal = pos % PAGE`，最终：

- K cache 布局 `[num_blocks, kv_heads, head_size//x, block, x]`——把 head 维拆成
  `d//x × x` 两级，使任意页内偏移的读都是 x 元素宽的合并访问；
- V cache 布局 `[num_blocks, kv_heads, head_size, block]`（转置布局，方便
  `p @ v` 时按列连续读）。

**所有 stride 显式传参、全程零 `.contiguous()`**——这正是我们 P2 的
TritonRoPE 里用 contiguous/clone 兜底所回避掉的问题的工业解法。

## 3. online softmax（flash 流式归约）

标准 flash attention 模式：维护行级运行最大值 `m_i` 与指数和 `l_i`，每块算完
`alpha = exp(m_i - m_ij)` 把累积输出 `acc` 缩放后累加新块。值得抄的防御：

- `alpha = tl.where(m_i == -inf, 0.0, alpha)`——整行被 mask 光时（sliding window
  常见）避免 `exp(-inf - -inf) = NaN`。我们在 P1 手写 attention 时没踩到只是
  因为没有滑动窗口。
- 末尾 `acc / (l_i + 1e-10)` 防 0 除。

## 4. 两段循环 = chunked prefill 的注意力语义

先跑 **context 段**（历史 KV，无 causal mask——历史对当前全可见），再跑
**当前 chunk 段**（query 之间的 causal mask）。这把"增量 prefill"的语义直接
写进了 kernel 的控制流，而不是靠上层拼。

## 5. 其他可借鉴技巧（≥3 个，照单全收）

1. `tl.multiple_of(start_n, BLOCK_N)`——给编译器的对齐提示，帮助向量化。
2. `tl.range(..., loop_unroll_factor=4)`——手动控制软件流水展开度。
3. 满块快路径：`if 尾块 or 有 padding: masked load else: 无 mask load`——
   大多数块免掉 mask 计算。
4. `BLOCK_DMODEL_PADDED`（2 的幂）+ `dim_mask`——处理非 2 幂 head_dim 的
   通用手法（我们 P2 的 softmax wrapper 也在用同一招）。
5. 平台分支收敛在模块顶部常量（`NUM_WARPS = 4 if is_rocm() else 8`、
   sm80 以下 BASE_BLOCK=64——我们的 gfx1201 同样适用 64/4 这一档）。
6. fp8 路径全部用 `tl.load(scale)` 在 kernel 内反量化，scale 本身是 GPU 张量
   而非标量参数——支持 per-tensor 动态 scale。

## 6. 对照我们的实现

| 主题 | vLLM 做法 | 我们 P1/P2 现状 | P3 计划 |
|---|---|---|---|
| KV 寻址 | B_Loc + 显式 stride | contiguous KV + read permute | 采纳 B_Loc 模式 |
| 非连续输入 | 显式 stride 索引 | contiguous/clone 兜底（见 rope.py 踩坑注释） | decode kernel 直接收 stride |
| softmax | online（flash 流式） | 两步 materialize（scores→softmax→PV） | decode 单 query 场景无需 online；prefill 侧可选进阶 |
| 头维 padding | PADDED + dim_mask | BLOCK=next_pow2 + mask（同思路） | 沿用 |

一句话总结：这个 kernel 把"分页 KV 的地址换算"做成了独立的 `@triton.jit`
辅助函数（`_paged_kv_cache_offsets`），控制流就是 chunked prefill 的语义本身，
防御性 `where` 和满块快路径值得逐条抄进 P3 的实现清单。
