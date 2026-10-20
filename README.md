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
| **P8 加固（mark_static_address）** | **220** | **1738（B=16 3208，B=32 5023）** |
| llama.cpp（同卡同模型 bf16，参照） | 213.4 | —（未测） |
| 实测带宽上限（纯读 420GB/s） | — | ~3400（B=8 等效） |

（"理论上限"改用实测：标称 640GB/s 是纸面值，大块纯读实测 420GB/s，
1.19GB 权重 + 0.31GB lm_head → 每步 ~3.6ms 下限。B=32 的 5023 已显著
超过该"B=8 等效"值——批越大每 token 摊的权重读越薄，上限随 B 上移。）

下一步杠杆（按预期收益）：权重量化 int4/int8（权重读减 2-4x）、
投机解码、chunked prefill（长 prompt 分片防独占）、prefill 编译兼容。

