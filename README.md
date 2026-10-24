# torchinfra

PyTorch 之上的迷你 LLM 推理引擎与自定义算子学习栈（ROCm / Radeon RX 9070 GRE）。

设计文档见 [docs/superpowers/specs/2026-10-19-torchinfra-design.md](docs/superpowers/specs/2026-10-19-torchinfra-design.md)，
路线：P0 编译链路 → P1 纯 PyTorch 推理引擎 → P2 自定义算子 → P3 paged KV + continuous batching。

## 环境（P0 踩坑结论，换机器/重建 venv 必读）

- venv：`~/rocm10`（uv 创建，Python 3.14.7，ROCm 10.1 pip wheels + torch 2.14.0+rocm10.1.0）
- **必须先执行一次 `~/rocm10/bin/rocm-sdk init`**：展开 devel payload（thrust、hipblas
  等头文件）并把 device 文件链接进 devel 树。未展开时 JIT 扩展编译会报
  `thrust/complex.h file not found`；展开前 torch 把 ROCM_HOME 解析到
  `_rocm_sdk_core`，展开后正确解析到 `_rocm_sdk_devel`
- `PYTORCH_ROCM_ARCH=gfx1201` 由 `torchinfra/kernels/loader.py` 自动设置，
  无需手动 export
- loader 额外传了 `--rocm-device-lib-path=$ROCM_HOME/lib/llvm/amdgcn/bitcode`
  （pip wheel 布局下 clang 默认搜不到 device bitcode；devel 展开后理论上可选，
  保留它作为确定性保险）
- 编译缓存：`~/.cache/torch_extensions/`；改 kernel 源码或 flags 后自动重编

### 已知问题

编译失败时 torch 可能把真实错误吞成
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 ...`
（仓库路径含中文 + ninja 错误输出解码的 bug）。看真实错误的方法：

```bash
cd ~/.cache/torch_extensions/py314_rocm71626385/torchinfra_kernels
ninja            # 原始错误直接可见
```

## 运行测试

```bash
cd ~/项目/torchinfra
for t in tests/test_*.py; do PYTHONPATH=. ~/rocm10/bin/python $t; done
```

## 对话

```bash
cd ~/项目/torchinfra
PYTHONPATH=. ~/rocm10/bin/python -m torchinfra.chat   # 默认 Qwen3-0.6B bf16
```

参数：`--max-new/--temperature/--top-p/--seed`。实测 decode 约 14 tok/s（bf16，9070 GRE）。

## 当前状态

- [x] P0 — torch extension 编译链路调通，`torch.ops.torchinfra.vector_add`（bf16）可用
- [x] P1 — 纯 PyTorch 推理引擎（Qwen3-0.6B）：手写前向与 transformers 对拍
  top-1 一致率 0.943（bf16）；fp32 greedy 生成 32 token 与 HF **逐 token 一致**；
  chat CLI 可交互对话（14 tok/s）
- [x] P2 — 自定义算子替换（Triton：add / RMSNorm / softmax / RoPE + fused residual）
  全部对拍一致；引擎 `backend="triton"` 通过 fp32 greedy 验收。算子基准（9070 GRE，do_bench）：

  | 算子 | torch ms | triton ms | 加速比 |
  |---|---|---|---|
  | RMSNorm [1,4096,1024] bf16 | 0.289 | 0.067 | 4.28x |
  | softmax [32,4096] bf16 | 0.022 | 0.009 | 2.38x |
  | RoPE q16/k8×512×128 bf16 | 0.156 | 0.030 | 5.22x |
  | add 16M bf16 | 0.356 | 0.354 | 1.01x |
  | residual+RMSNorm fused | 0.328 | 0.130 | 2.52x |

  add 是带宽瓶颈算子（1.01x 说明 torch 已最优）；RMSNorm/RoPE 大幅领先是吃掉了
  torch eager 的小算子调度开销；fused 省一趟显存往返拿到 2.5x。
  Triton 踩坑记录：s==1 时 transposed 视图被 torch 视作"连续"但 stride(2)=H*D，
  kernel 按行主序寻址直接越界（GPU page fault），需 `clone(memory_format=
  torch.contiguous_format)` 强制紧凑布局（见 `triton_kernels/rope.py`）。
- [x] P3 — paged KV cache + Triton paged decode attention + continuous batching
  验收全过：4 请求并发输出与串行**逐 token 一致**（fp32）；吞吐 bf16 下
  8 请求并发 175 tok/s vs 串行 82.4 tok/s = **2.12x**。
  架构：block table 两级映射（vLLM B_Loc 模式）+ FlashDecoding 式两阶段
  attention（grid 展开 tile + log-sum-exp 合并，长上下文 decode 并行化的
  真实手法）+ prefill 优先的迭代级调度。
  **勘误（2026-10-19 复核）**：开发中一度误判"triton 3.8.0+gfx1201 存在地址
  多乘法编译 bug"。经独立最小复现（arange 数据指纹 + 对照矩阵）与"循环版
  kernel + 正确局部行号"复核证伪——triton 无此 bug，真实原因是自己的地址
  bug（全局 token 序号误作块内行号）；调试中的"地址存出来正确"探针又因期望值
  用了同样的错误算法而成为循环论证。教训：对拍参考必须与被测代码独立推导，
  否则只是自己证明自己。
  **后续修正**：假 bug 期间的 FlashDecoding 两阶段重构被 A/B 实测证伪性能
  （scratch 往返 + torch 合并开销，慢 1.5-4.3x，见 `tools/bench_paged_attn.py`），
  默认 kernel 已切回单 kernel 循环版（`paged_decode_attention`），两阶段版
  保留为 `paged_decode_attention_splitk` 供对比；端到端并发吞吐 175 → 187.9
  tok/s（2.28x）。
- [x] P4 — decode 的 CUDA Graph 化（vLLM 式三件套）
  ① cache 写入 GPU 化：`append_kv` kernel 消灭逐请求 `int()` 同步点
  （`torch.compile` 实测因此于 decode 慢 0.74x）；② 静态缓冲 + pad 到固定
  batch，pad 槽位指向预留 dummy 块 0；③ `torch.cuda.CUDAGraph` 捕获整个
  decode step。rope/append 读 `lens`、attention 读独立缓冲 `lens_p1`（图内
  零原地自增，避免 28 层累加）。
  实测（bf16）：decode 单步 B=8 33.0 → 8.3ms（**3.9x**，launch 开销占 eager
  75%）；端到端并发吞吐 187.9 → **423.9 tok/s（vs 串行 5.16x）**；
  并发==串行逐 token 验收在图路径下依旧通过。prefill 侧
  `from_pretrained(compile=True)` 1.88x（torch.compile，decode 不适用）。
  **piecewise 升级（默认 `graph_mode="piecewise"`）**：rope/append/attention
  注册为正式 custom op（带 `mutates_args`），`torch.compile(mode=
  "reduce-overhead")`（Inductor cudagraph trees）按 op 边界切图——在整图
  消除 launch 开销之上叠加 Inductor 的 norm/proj/MLP 融合。稳态单步
  8.19 → 6.03ms（**再 1.36x**），端到端并发吞吐 **502.2 tok/s（vs 串行
  5.80x）**。注意一次性成本：首次调用含 Inductor 编译约数秒（真实服务在
  启动时预热，短跑基准需摊平或剔除）。`graph_mode` 可选 piecewise /
  manual（整图捕获）/ none（eager）。
- [x] P5 — decode CPU/带宽优化（对照 llama.cpp 实测追赶）
  同硬件同模型同精度基线（llama-bench，本机 build-amd）：**单流 213.4 tok/s**。
  四项优化：批量 GPU 采样（单次同步）、请求长度纯 python 记账（零 GPU 读回）、
  多档 batch 桶 1/2/4/8（单流不再陪跑 dummy）、QKV 三合一 + gate/up 二合一
  投影融合（每层 6 个瘦 GEMM → 2 个，`warmup()` 启动预热消除编译税）。
  **单流 61 → 177 tok/s（llama.cpp 的 83%）；并发 502 → 1208 tok/s
  （vs 串行 13.9x）**。并修复 block_table 行数按 rid 索引的越界 bug
  （rid 单调递增不复用，行数须覆盖生命周期总请求数）。
- [x] P6 — 零同步 decode + kernel 融合（不量化的最后一轮榨取）
  先剖析后动手（tools/profile_decode.py）：B=8 每步 wall 6.39ms 中纯 GPU
  4.80ms，CPU 串行 1.22ms；GPU 内 hipblas 瘦 GEMV 占 72%（实测 345GB/s）。
  ① **零同步稳态循环**（vLLM v1 overlap scheduling 核心）：热循环仅
  `advance`（lens+1 一次 kernel）→ replay → GPU 采样（greedy=argmax）→
  tokens GPU 直填 → 双缓冲 pinned 异步拷贝 + event，完成检测滞后消费
  （深度 2：`query()` 未就绪缓一步，复写前才强制同步）——CPU 时间被 GPU
  完全遮蔽；staging 只在组合变化时发生（bt 行 reserve 后不变）。
  ② **QK-norm + RoPE + cache 写入三合一 kernel**（fused_decode.py，
  piecewise 下注册为 fused_qkrope_op）：每层省 ~6 个小 kernel 与 rope_op
  的两次 copy_（实测每层 4.0us，替代 ~0.3ms/步的小 kernel 群）。
  ③ 桶扩到 16/32：权重扫描 1.19GB/步是固定成本，B 越大摊越薄。
  ④ attention `num_warps` 4→2（实测 B=8 -11%、B=32 -23%）。
  **单流 177 → 221 tok/s（llama.cpp 的 103%，反超）；B=8 并发 1208 →
  1493（vs 串行 17.5x）；新增 B=16 2460、B=32 3405 tok/s**。
  诚实记录两条"试过但不行"：Triton 定制瘦 GEMV 全面输给 hipblas
  （170-360 vs 270-400GB/s，tools/bench_gemv.py，需 L2 轮换权重才测得真）；
  编译版 prefill 与 piecewise cudagraph trees 冲突（prefill 插在两次回放
  之间触发 Inductor 跳图，decode 退化 2.3x）。教训一条：差分测试的
  cache 必须多层且用非 0 层索引——单层 cache 恰好掩盖了层偏移 bug
  （kernel 写全落第 0 层，q 却全对，端到端第 2 个 token 才露馅）。
  **B=32 时瓶颈已翻转到 prefill**（32 个 eager prefill 串行 ~1.6s，
  占墙钟 2/3；decode 稳态仅 ~3.2ms/步）。
- [x] P7 — 批量 prefill（打包前向 + varlen 因果 kernel）
  消灭 P6 发现的 prefill 瓶颈：同一轮装载的全部 prompt 打包成**一次**前向
  （prefill.py）——权重部分全批化（embedding/qkv/o/MLP/lm_head 都在
  [sum_S, E] 上跑，32 个 prompt 只扫一遍权重）；attention 用 Triton varlen
  因果 kernel（cu_seqlens 划界 + 在线 softmax，prefill 不读 cache——上下文
  就是本轮自己算出的 k/v）；k/v 写入 paged cache 用打包 kernel（块表寻址，
  grid (B, n_kv, maxS)）。RoPE/QK-norm 与 apply_qk/RMSNorm 逐位同式。
  差分测试（test_prefill.py，长度含 16/17/33 块边界）：logits max diff
  2.2e-05、argmax 全一致、cache 逐层对拍通过。开发中踩两个坑记录在案：
  kernel 实参顺序传反（cache 指针与打包指针互换 → 页错误）；acc*alpha
  少广播维（2D acc 需 alpha[:, None]）。
  **B=8 1493 → 1707（+14%）；B=16 2460 → 3160（+28%）；B=32 3405 →
  4980 tok/s（+46%）**；单流 217（llama.cpp 的 102%）持平——decode 端
  未动。旧逐请求 prefill 与其调度器死代码已删除。

- [x] P8 — 对照 vLLM/SGLang 源码的一轮加固（一个雷、一个崩、一个负结果）
  通读 vLLM `config/compilation.py` 与 SGLang 调度/图实现后的三件事：
  ① **修复潜伏两阶段的静默退化雷**：`fused_qkrope_op` 对 KV cache 的
  mutation 会让 cudagraph trees 报 "skipping cudagraphs due to mutated
  inputs"——P6/P7 的图一直在旧编译缓存的历史路径上跑，某次核心转储
  触发重编译后 decode 从 1598 **静默跌到 283 tok/s（5.6x）**。修法即
  vLLM 同款：`torch._dynamo.mark_static_address(k/v cache)`（cache 地址
  与引擎同生命周期，不迁移）。
  ② **修复 unfused 回退 + ①的崩溃**：runners 未融合时 `rope_op` 原地改
  q/k，配合 mark_static_address 会让 Inductor 做整 cache 函数化克隆
  （grid 26 万的 pointwise）→ 页错误。两个 runner 构造时统一自动
  `fuse_decode_projections`（幂等）。
  ③ **负结果**：按 vLLM FULL 模式思路用 `fused_rmsnorm_residual` 重写
  manual 整图链（每层 9 kernel）并回退——gfx1201 上 `_fused_kernel` 在
  深层 eager 链中表现出「输入已验证正确、输出连 x+dx 都错」且随脚本
  结构（函数级/模块级）翻转的行为；25+ 次取证（多进程重放/克隆输入/
  同步/清缓存/逐部件排除 silu·attention·eps·别名）未定位根因，单测
  始终全绿、仅长链触发。保守回退，default piecewise 不受影响。
  复测（修雷后图恢复+稳定捕获）：单流 220 / B=8 1738 / B=16 3208 /
  **B=32 5023 tok/s**（18.5x），全部 21 测试通过。
  调研中其它值得记录的：vLLM 默认 FULL_AND_PIECEWISE（decode 整图），
  有与我们 P6 同款的 `fuse_qk_norm_rope_kvcache` pass（方向印证）；
  SGLang 的 overlap 调度我们在 P6 已实现同构；两大引擎的下一个大
  杠杆是 n-gram/EAGLE 投机解码（无需量化、拒绝采样保持分布不变）。
- [x] P9 — prefill 榨干（decode 贴带宽了，prefill 还远没有）
  追问之下剖析（tools/profile_prefill.py）：prefill 是**计算**瓶颈，
  77.3ms/416token ≈ 7 TFLOPS，其中 `_prefill_attn_kernel` 独占 74.7%——
  初版把 q/k/v 全升 fp32 再 `tl.dot`，在 RDNA4 上不走 tensor core，
  仅 ~0.3 TFLOPS。两步修复：
  ① attention dot 改 bf16 原生输入 + fp32 累加（tensor core 路径），
  47.8 → 1.3ms（单点 38x），差分不变（bf16 输入本来就是 bf16 值，
  tensor core 乘积精确，累加同为 fp32）。
  ② 每层 ~18 个 eager pointwise（qk-norm 的 float/pow/mean/rsqrt 链 +
  rope 的 cat/mul + 打包 cache 写入）→ **复用 decode 的三合一 kernel**：
  把 sum_S 当 batch 维、逐 token 位置当 lens、块表按行 gather，
  `fused_qk_rope_append(..., want_k=True)` 一趟完成（kernel 加 constexpr
  KOut 输出，decode 路径零改动）。
  结果：B=32 批量 prefill **77 → 16.4ms（4.7x）**；单条 prefill 50 →
  8.7ms（TTFT）。端到端：B=32 5023 → **5265 tok/s**，B=16 3208 → 3317，
  B=8 1738 → 1772，单流 220 → 223（llama.cpp 的 104%）。
  **勘误（对照官方规格实测后修正）**：本卡为 12GB/192-bit 版，规格带宽
  432GB/s——实测纯读 420 = **97%，decode 确实贴墙**。但算力没贴满：
  规格矩阵 FP16 峰值 137 TFLOPS，实测方阵 GEMM 上限 ~104 TFLOPS；
  prefill 端到端仅 22（M=416）~37（M≥1024）TFLOPS（GEMM 部分 ~39），
  约为可达成算力的 1/3——**prefill 仍有 ~2.5x GEMM 效率空间**（M=8 时
  输给 hipblas 的结论不适用于 M≥416 的计算型形状）。短 prompt 负载下
  prefill 只占端到端 ~2% 故未继续追；长 prompt TTFT 场景值得回来挖。

- [x] P10 — 长输入 prefill 拉满（13 token 短 prompt 之外的真实战场）
  用户点破"P9 拿 13 token 测的不算数"。长输入基准确认：S=4096 时注意力
  kernel 占 50%（23.8 TFLOPS），S=8192 端到端只剩 14.3 TFLOPS（二次项
  吃掉一半算力）。四步修复：
  ① **注意力换 SDPA**（max_s≥512 时逐请求走 torch 融合后端
  `flyc_attn_fwd`：S=4096 实测 67.6 vs 自研 kernel 调优后 25.3 TFLOPS，
  2.7x——短批仍走自研打包 kernel，两者混合）；
  ② 自研 kernel tile 调优备胎（TQ=128/TK=64/w8/st2，64KB LDS 上限内）；
  ③ qk-norm+rope+cache 三合一改 **2D 行块 tile**（[16 行,128 维] 向量化、
  归约沿 D 轴，cos/sin 行内复用；行循环版 S=4096 实测 7.9ms→tile 版
  ~4ms）。教训一条：**TR=8 配置在 gfx1201 上确定性算错**（TR=16 正确、
  内部行也错排除掩码问题）——又一个 codegen 怪癖，锁定 TR=16；
  ④ norm/silu 换 P2 稳定 Triton kernel（`triton_rmsnorm`/`silu_mul`，
  避开 P8 出问题的 `_fused_kernel`）+ prefill_batch 自动投影融合
  （未融合时会走 cat 回退慢 30%+）。
  结果（tools/bench_prefill.py）：S=4096 单条 **165 → 96ms（-42%）**、
  S=8192 **503 → 257ms（-49%）**、S=1024 29.6→18.9、B=8×128 25.1→18.7。
  机器实际吞吐（含注意力 FLOPs）S=4096 ≈ **57 TFLOPS**（可达成上限
  104 的 55%；剩余大头 GEMM@74 与 SDPA@67 均已贴各自天花板）。
  官方吞吐测试补 prefill 预热后 **19.5x / 1686 tok/s**（B=8）。端到端
  短 prompt 基准不变（B=32 5240）。差分测试补长序列组（600/512/700）
  覆盖 SDPA 路径，21 测试全绿。

- [x] P13 — Nemotron-3.5-Rerank-8B（ministral3 双向塔）移植 fp8+Triton，
  32K 全长可用（详见文末 P13 节）

## 性能演进总表（9070 GRE，Qwen3-0.6B bf16）

| 阶段 | 单流 tok/s | 并发 B=8 tok/s |
|---|---|---|
| P1 手写引擎（eager chat） | ~14 | — |
| P3 分页 + continuous batching | 61–83 | 188 |
| P4 CUDA Graph 整图 | 61 | 424 |
| P4 piecewise（Inductor 融合） | 61 | 502 |
| P5 批采样/记账/桶/投影融合 | 177 | 1208 |
| P6 零同步 + kernel 融合 | 221 | 1493（B=16 2460，B=32 3405） |
| P7 批量 prefill | 217 | 1707（B=16 3160，B=32 4980） |
| P8 加固（mark_static_address） | 220 | 1738（B=16 3208，B=32 5023） |
| P9 prefill 榨干（短 prompt） | 223 | 1772（B=16 3317，B=32 5265） |
| **P10 长输入 prefill（SDPA 混合 + tile 化）** | **224** | **1686 官方 19.5x（B=32 5240；长输入 S=4K 减半）** |
| llama.cpp（同卡同模型 bf16，参照） | 213.4 | —（未测） |
| 实测带宽上限（纯读 420 / 规格 432GB/s） | — | ~3400（B=8 等效） |

（"理论上限"改用实测：标称 640GB/s 是纸面值，大块纯读实测 420GB/s，
1.19GB 权重 + 0.31GB lm_head → 每步 ~3.6ms 下限（实测纯读 420GB/s =
规格 432 的 97%，decode 贴墙）。B=32 的 5265 已显著超过该"B=8 等效"
值——批越大每 token 摊的权重读越薄，上限随 B 上移。算力侧（勘误后）：
规格矩阵峰值 137、方阵实测 ~104，prefill 端到端 22-37——未贴满。）

## P11 — jina-reranker-v3.5 重排路径（Qwen3-0.6B 骨干 + 混合滑窗 + FP8）

jina-reranker-v3.5 = Qwen3-0.6B 骨干 + 16/28 层滑窗注意力 + projector 余弦打分。
`torchinfra/jina_rerank.py` 在引擎之上自研前向（注意力走 SDPA / 自研滑窗 kernel，
norm/QKV/MLP 走 P2 Triton 融合 + tensorwise E4M3 权重/激活量化），`rerank_server.py`
提供 vLLM 风格 `/v1/rerank` 并发服务（调度器多请求块拼批）。对拍
（`tests/test_jina_rerank.py`，vs 官方 HF forward）：分数误差 < 0.005。

本轮 e2e 压延迟（tools/bench_rerank_e2e.py，5 档负载，fp8）：

| 负载 | 基线 e2e | 本轮 | 变化 |
|---|---|---|---|
| 8×64 文档（S=767） | 16.3 ms | 15.2 ms | -7% |
| 32×256（S=8.8K） | 230 ms | 200 ms | -13% |
| 100×256（S=27.1K） | 1098 ms | 995 ms | -9% |
| 20×512（S=10.7K） | 296 ms | 259 ms | -12% |
| 200×512（S=105.5K/2块） | 6572 ms | 6172 ms | -6% |

四项改动与三个负结果/踩坑（全部实录）：

1. **修复 >32K 序列 GPU 页错误**：`plan_blocks` 按 tokenizer.model_max_length
   （131072）切块，但 RoPE 表曾被硬限 32768——超长块越界读 cos/sin 表直接
   `HSA_STATUS_ERROR_MEMORY_FAULT`（队列死、进程挂）。表放宽到
   cfg.max_position_embeddings（~134MB），另加显式长度守卫。
2. **滑窗注意力 kernel 重写**（三段式 FA2 循环切分 + [B,S,H*D] 直写）：
   远端段只保留窗掩码、中段满块免掩码、近对角段只保留因果掩码，输出直写
   行主序省掉滑窗层的 transpose+contiguous。S=27K 单层 8.12→4.01ms
   （57 TFLOPS，2x）。**踩坑**：首版三段边界用理想区间值，段 2 末块越过界、
   段 3 从界值起步 → 重叠 k 双计（对拍差 0.06 暴露）；改为三段共用锚定
   start 的 TK 网格后对拍干净（含 B=2 左 padding、S 跨 1024/2048 边界组）。
   配置扫描（tools/bench_window_cfg.py）：64/64/w4/st1 最优，st2/大 tile
   在 gfx1201 上慢 2x（与 P6 结论一致）。
3. **qk-norm+RoPE kernel 兼写 v**：v 原为 qkv 切片后 permute+contiguous
   （每层一次 transpose 型拷贝，~170GB/s）；改为在三合一 kernel 里顺带搬运
   （WANT_V），实测 kernel 总耗时不变（带宽有余量），每层净省 ~0.7ms@S=27K
   + 28 次 launch。TR=16/w8 略优。**插曲**：扫描脚本的 torch 参考把
   RMSNorm 分母写成 64（(a²+b²) 逐元素相加后 mean(-1)），一度冤枉 kernel
   错 √2——新旧 kernel 逐位一致，手算才定位是参考错。
4. **整前向 CUDA Graph 分桶捕获（≤8192）**：右 padding + 静态 buffer 懒捕获
   （tools/test_graph_path.py 对拍）。两个坑：① `bool(valid.all())` 主机同步
   会使捕获直接作废（也挡 eager 流水）——改为调用方 CPU 侧判定传入；
   ② **右 padding 行不清零会污染 tensorwise fp8 量化的 amax**（pad 行激活
   改变整批 scale）——图内嵌入后按 rowmask 清零，零行对 amax 无贡献且
   逐行算子下恒为零。真实文本下图 vs eager **逐位一致**；随机 id 会放大
   跨 GEMM 算法档的 ~3e-3 舍入差（病态激活下两层放大到 64，真实负载无此
   现象）。**负结果（诚实记录）**：纯 replay 13.3ms ≈ eager 13.8ms@S=767
   ——小 S 档的瓶颈不是 CPU launch 而是 GPU 侧 kernel 派发间隔（~620 个
   kernel × ~8µs，图消不掉），下一步杠杆是减 kernel 数（量化 amax/quant
   融合进 rmsnorm/silu 生产者）而非图。

大 S 档（>27K 单块）前向里全注意力 SDPA(flyc) 占 ~50% 且已贴该 kernel
天花板（~67 TFLOPS），滑窗层优化后剩余空间在 GEMM（fp8 ~107 TFLOPS）与
kernel 数；200×512 的 CPU 侧（plan+encode ~196ms）仅占 3%，不构成瓶颈。

下一步杠杆（引擎侧，按预期收益）：权重量化 int4/int8（权重读减 2-4x）、
投机解码、chunked prefill（长 prompt 分片防独占）、prefill 编译兼容。
rerank 侧：fp8 量化链已融合到位（P12，生产者全部直吐 amax partial），
剩余结构性杠杆是权重 int4/int8 与注意力 fp8，均属大改。

**P12 — fp8 量化融合进生产者 kernel（2026-10-24）**：tensorwise 量化的
两阶段（先 amax 后 quant）没法一遍出 fp8，但 amax 那趟全量读可以白拿——
`_rmsnorm_amax_kernel` / `_silu_mul_amax_kernel` 在写 bf16 输出的同一趟
里把每行 amax 顺带写进 [M] fp32 partial（值本来就在寄存器里），新增
`fp8_linear_q(partial)` 跳过独立 `_row_amax` kernel，每层省 3 launch + 3 趟
激活读（down 输入 [27K,3072] 一趟就是 167MB）。第二轮把 o_proj 输入也收掉：
SDPA 输出的 transpose+contiguous 拷贝与 amax 合成 `_transpose_amax_kernel`
（转置同趟吐行 amax），滑窗 kernel 写 out 同趟吐 tile amax partial
（掩码行记 0；全掩码 tile 的 acc/l_i=NaN 必须 where 掉再归约）。两轮后
`_row_amax` 调用归零。**数值**：partial 一律取 bf16 舍入后的值（量化
kernel 读回的同一 bit pattern），amax 与归约顺序无关 → partial 与全量
行 amax 逐位一致、q 逐位一致（tools/test_fused_quant_parity.py 全 0，
含 transpose_amax、滑窗 partial、左 padding 组）。e2e（同 bench，两轮
累计 vs P11）：8×64 15.2→14.4ms、32×256 200→194.1、100×256 995→973、
20×512 259→252.2、200×512 6172→6075（-28ms，与每层省一趟 attn_out
全量读 55.6MB@27K 的估算吻合）。**踩坑**：`_rmsnorm_amax_kernel` 首版
抄了 `@triton.autotune`——BLOCK 影响平方和归约树，autotune 按计时选
配置会让输出随噪声漂移最后一位 ulp（微验证两个 M 档 q 差 2 位暴露），
bit-exact 关键 kernel 必须**锁死配置**（N=1024 单次迭代，BLOCK=1024/w4）。
**门禁翻新**：test_graph_path 原来拿随机 id 对拍 graph vs eager，fp8 下
跨桶 M 命中不同 GEMM 算法档（~3e-3/层被病态激活放大到 ~1e-1，TunableOp
CSV 追加调优行也会让结果漂移）——干净 HEAD 同样挂，属门禁误设计而非
图错误。改为两层：桶精确 S 必须 bit-exact（真正的捕获 bug 检测器，
S=256/1024/4096/8192 全 0），跨桶真实文本只约束投影归一化后的分数空间
（<0.02，实测 ~0.008）。**负结果**：rowwise scale（一行一 scale，生产者
可一遍出 fp8）在 `_scaled_mm` 上慢 20-35%（141→117 TF @M=27K，M=767 时
135→100），融合方案判死、走 tensorwise+partial（tools/bench_smm_scaling.py）。
另：CUPTI 在此栈上丢事件严重（CUDA 侧 kernel 计数不可信），kernel 数
主张只报结构性事实（_row_amax 28→0/前向），量化收益以 e2e 为准。

### serving：标准 rerank API（0.0.0.0）

    cd ~/项目/torchinfra
    PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RESULT_FILE=./tunableop_results0.csv \
    PYTHONPATH=. nohup ~/rocm10/bin/python -m torchinfra.rerank_server \
        --host 0.0.0.0 --port 8000 --model ~/models/jina-reranker-v3.5-fp8 &

对外是 Jina/Cohere/vLLM 通用的 `/v1/rerank`（另 `/rerank`、`/health`、
`/v1/models`）：

    POST /v1/rerank
    {"model": "jina-reranker-v3.5", "query": "...",
     "documents": ["...", ...], "top_n": 3, "return_documents": true}
    → {"results": [{"index": 0, "relevance_score": 0.35,
                    "document": {"text": "..."}}], "usage": {...}}

documents 也接受 `[{"text": "..."}]`；>125 篇自动分块加权合并；并发请求由
调度器拼批（`--max-batch`/`--coalesce-ms`，GPU 只有调度线程跑前向）。
**拼批按真实 token 长度分组**（rerank_server._run 两遍编码）：小块（≤512）
相近长度左 padding 拼批；大块只与等长块拼（无 padding 保 flash/is_causal
路径），否则单独 B=1 前向吃 CUDA graph 分桶——异长左 padding 在大 S 下
是双重毒药：小块被撑到大块长度白算十几倍，且 padded 批的 SDPA 掉出
flash、物化 B×H×S² 注意力矩阵（B=2、S≈2900 实测慢 7 倍，更大 S 直接
OOM；修复前 130 篇 453ms → 修复后 64ms）。

实测热态吞吐（fp8，含 HTTP/调度，0.0.0.0 绑定下局域网 IP 直打）：单请求
100 篇（2.2K tok）47ms ≈ **48K tok/s**；130 篇多块（3K tok）64ms ≈
**46K tok/s**；8 并发小请求 27-36ms/批 ≈ **45-60K tok/s**；混合
（2×130 篇 + 2×100 篇 + 4 小）212ms ≈ **52K tok/s**。首请求含 Triton JIT
与各长度桶的图捕获（首轮慢 3-10 倍属正常）。注意无鉴权，暴露到局域网
（含 tailscale 侧）需自行加防火墙规则。

**torch.compile 负结果（2026-10-23 实测，tools/bench_compile_rerank.py）**：
对 reranker 前向直接 compile 不可用——S=767 能编译但**慢于 eager**
（15.3 vs 13.9ms，0.91x）且**数值错**（hidden 相对误差 9.2e-2，P8
_fused_kernel 深层链翻转 bug 同款现象，长链触发、对拍拦截）；S=8782
直接编译失败（Inductor 子进程无法 build 被 dynamo 追踪的
`_window_attn_kernel`，三段循环的运行期边界过不了它的 Triton codegen）。
结论：该前向的 kernel 数问题只能靠手工融合（同 P8 的 custom-op 边界
哲学），compile 路线在此栈（gfx1201 + 裸 triton kernel + fp8）判死。


## P13 — Nemotron-3.5-Rerank-8B 移植（ministral3 双向塔 + FP8 + 32K，2026-10-24）

`torchinfra/nemotron_rerank.py`：NVIDIA Nemotron-3.5-Rerank-8B（Mistral3
序列分类壳 + ministral3 文本塔，34 层 / hidden 4096 / 32Q8KV×128 /
SwiGLU 14336）从 transformers 5.12.1 逐条移植到 torchinfra fp8 栈。
架构要点（与 jina 骨干的三处关键差异）：**全双向注意力**（无 is_causal）；
**无 qk-norm**；yarn rope（factor 16 / base 1e6 / orig 16K，float32 全程
逐位对齐 `_compute_yarn_parameters`）+ llama-4 Q attn-scale（pos<16K 恒 1，
kernel 用 constexpr 关掉；16K–32K 段 ≈1.069 只乘 Q）。打分 = final
RMSNorm → mask mean-pool → fp32 Linear(4096→1)。

性能设计与实测（B=1，`tools/bench_nemotron.py`）：

| S | 路径 | 延迟 | tok/s | 峰值显存 |
|---|---|---|---|---|
| 512 | CUDA graph | 64.5ms | 7936 | 8.31 GiB |
| 2048 | CUDA graph | 258.7ms | 7916 | 8.39 GiB |
| 8192 | SDPA flash | 1428ms | 5736 | 9.51 GiB |
| 16384 | SDPA flash | 3932ms | 4167 | 10.07 GiB |
| 32768 | SDPA flash | 12182ms | 2690 | 10.51 GiB |

B=8×2048 掩码批（自研双向 kernel）2141ms ≈ 7651 tok/s，只比同长 flash
图路径慢 3%。权重 7.98GiB（fp8 8.8GB checkpoint，QKV/gate-up 融合单
GEMM），**32K 全长在本卡成立**（12GB 卡实测峰值 10.5GiB）。

kernel 组成（榨性能的取舍全部有实测依据）：GEMM 清一色
`torch._scaled_mm` tensorwise fp8（P12 生产者 amax partial 直喂）；
无掩码注意力走 SDPA flash（此卡天花板，S=32K 实测 146.5TF）；**新写两个
Triton kernel**：① `_rope_qkv_kernel`（yarn rope + llama4 scale +
[B*S,6144]→[B,H,S,D] 转置一趟，q/k/v 三份输出直出 SDPA 布局）；
② `_bidir_attn_kernel`（FA2 式在线 softmax 双向注意力，右 padding vlen
掩码，pad 行写零防污染 tensorwise amax，逐 tile amax partial 直喂
o_proj 量化）——存在理由：**ROCm 上 SDPA 带 attn_mask 时 flash 与
efficient 双双 "No available kernel"，掉 math 后端物化 [B,H,S,S]**
（B=2 S=3K 就是 7 倍慢，32K 直接 OOM），所以凡掩码路径必须自研。
MLP 在 B*S>8192 时沿行分块（gu [S,28672] 的 1.9GB 峰值压到 448MB）；
B=1 且 S≤2048 走 CUDA graph 分桶（右 padding 256 对齐，vlen 走
static tensor replay 时改写），更大 S 图收益抵不过自研 kernel 与
flash 的差距，走 eager flash。

精度（`tools/test_nemotron_e2e.py`，vs CPU bf16 transformers 参考 8 对）：
max|Δ|=0.091（阈值 0.3），**打分排序全保**；等长 batch 逐位一致；
右 padding batch vs 单条 Δ≤0.025——非 bug，是 tensorwise fp8 固有
（整批一个 scale，批组成变则同伴文档的量化步长跟着变），各自离 bf16
参考仍 <0.05。yarn 表 float32 逐位对齐 transformers（float64 反而有
6e-8 偏差）。参考生成 `tools/nemotron_cpu_ref.py`（CPU bf16 eager）。

**踩坑两记（本轮最大成本，都值得进族谱）**：
① **stride(0) 语义陷阱**：rope kernel 按 [B*S, C] 展平行寻址，调用处传
了三维 qkv 的 `stride(0)`（=S·C≈1.5M 而非 C=6144）——二维单元测试里
恰好等于 C，掩盖了 bug；真实三维跑直接越界读出
HSA_STATUS_ERROR_MEMORY_FAULT。**故障表现极具欺骗性**：队列被打挂后
`synchronize()` 在 CPU 上单核 100% 死等，GPU 空闲风扇不转，看起来像
"调优在 CPU 上转"——曾误诊为 TunableOp 现场 tuning 并为此关了
TUNING=1（该改动保留：调优 kernel 混进图捕获同样会打挂队列）。
真定位靠 `HIP_LAUNCH_BLOCKING=1` 前台探针逐 kernel 打点（/tmp/probe
式），错误当场在 launch 点爆出 kernel 名。教训：**单元测试的张量形状
必须和生产完全一致（含维数），否则 stride 类参数验证是空谈**。
② **32K OOM 的真凶是分配器碎片**：权重+工作集本可塞下（10.5<11.9GiB），
但 qkv 384MB / gu 448MB 大块反复进出后固定尺寸缓存器留了 ~900MB
reserved-unallocated 碎片。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
（模块头 setdefault）+ rope 后立刻 `del qkv` 后 32K 稳定通过。

### P13.1 — per-projection scale：融合 GEMM 不拆，段修正乘进下游 kernel（2026-10-24）

对 P13 的 QKV/gate-up 共享 scale 做精度升级。**反直觉的实验发现先行**：
逐张量 dequant 余弦几乎不随 scale 拆分变化（q/k/v 各投影 ±1e-5，V 的
scale 细了 3.2 倍余弦纹丝不动）——fp8 是浮点格式，**换 scale 不改变每个
元素的相对量化精度**（e4m3 相对网格密度与 scale 无关，只在裁剪/下溢
边缘起作用，权重分布良好到不了边缘）。scale 粒度真正要紧的是 INT 格式
和有离群通道的激活。共享版真正的损伤源是**共享 scale 让 q/k/v 的量化
误差结构相关**，穿过注意力复利放大；拆开后误差独立化，端到端
**max|Δ| 0.0914 → 0.0403**（同 bf16 参考 8 对，排序全保）。

三种落地方案的实测对比（数学上都等价于 per-column epilogue descale）：

| 方案 | GEMM 代价 | 确定性 | 结论 |
|---|---|---|---|
| rowwise scale_b [1,N]（_scaled_mm 原生） | 1.66×（0.683→1.136ms@qkv 形状） | — | 判死 |
| 拆 3+2 个 tensorwise GEMM（fp8_linear_q_multi，激活只量化一次） | 实测 ±0（N=1024 瘦 GEMM 也没吃亏） | 瘦形状触发 hipBLASLt split-k 原子归约，批内相同行抖 6.6e-4 | 可用但非终态 |
| **融合单 GEMM + 下游 kernel 段修正（采用）** | 0（就是原融合 GEMM） | 逐位确定 | 终态 |

采用方案的机制：q/k/v 仍拼成 [6144,4096] 单 GEMM，各段按自己的 scale
量化；GEMM 的 tensorwise scale_b 取 q 段的 sq，k/v 段的比值修正
(kc=sk/sq, vc=sv/sq) 由 `_rope_qkv_kernel` 在 load 后 fp32 补乘（它本来
就要读 k/v 做 rope，零额外访存）；gate/up 同理，修正乘进 silu_mul_amax。
数学上等价于 GEMM epilogue 按列 descale（激活共用量化也对——同一 X 进
qkv/gu，一个 tensorwise 激活 scale 本来就只算一次），但不需要手写 GEMM，
也不掉进慢 1.66× 的 rowwise 路径。共享 scale checkpoint 的 kc=vc=uc≡1，
同一 loader 兼容加载（对拍 0.0914 与旧版逐位一致）。

量化侧：`save_fp8_checkpoint_sharded(..., split_scales=True)`（盘上 key
布局不变，config 标 `tensor_per_projection`）；产物
`~/models/Nemotron-3.5-Rerank-8B-FP8S`（推荐，8.8GB，与 FP8 同尺寸）。
bench 与 P13 基线持平（S=512 64.3ms/7964 tok/s … 32K 12171ms），等长
batch 恢复逐位一致（Δ=0.00e+00）。

### serving：Nemotron 走同一套 /v1/rerank（P13.2，2026-10-24）

    PYTHONPATH=. nohup ~/rocm10/bin/python -m torchinfra.rerank_server \
        --host 0.0.0.0 --port 8000 \
        --model ~/models/Nemotron-3.5-Rerank-8B-FP8S &

`--backend auto` 按 config.json 的 architectures 判别（*
ForSequenceClassification → Nemotron pointwise，否则 jina bi-encoder），
HTTP 层完全复用；调度器按模型各有一套 `_run`：jina（bi-encoder，左
padding + embedding/分块合并）不动，新增 `NemotronScheduler`（pointwise
cross-encoder，**右 padding**）——复用队列/合并/两遍编码/长度分组全套
策略：小块（≤512）相近长度右 padding 拼批走自研双向掩码 kernel，大块
只与等长块拼（保 flash），单块 B=1 吃 CUDA graph 分桶；超 32K 的对由
`encode_pair_ids` 截断到 rope 表容量。返回原始 logit（Nemotron 官方
语义，非 0-1 分数，越大越相关）。

实测（FP8S，含 HTTP/调度/模板渲染）：100 篇中等（10.7K tok）1.37s ≈
**7.8K tok/s**；8 并发×10 篇 0.99s ≈ **8.7K tok/s**；15.7K tok 单文档
3.85s（与裸模型 bench 3927ms 吻合，serving 层零损耗）。与 jina（0.6B
bi-encoder，46-60K tok/s）同接口共存：8B cross-encoder 逐对精排代价
高 ~6 倍，换来的是交叉注意力精度档位——两者按场景选型即可。
