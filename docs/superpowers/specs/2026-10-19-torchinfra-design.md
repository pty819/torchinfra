# torchinfra 设计文档 — PyTorch 之上的迷你 LLM 推理引擎与自定义算子学习栈

- 日期：2026-10-19
- 状态：设计已评审通过，待编写实施计划
- 作者：pty819（与 ZCode 结对设计）

## 1. 背景与目标

在自有的 AMD Radeon RX 9070 GRE 上，按照 vLLM / SGLang 的真实构建方式（PyTorch 作底座 + 关键路径自定义算子 + 引擎层自管调度与显存），构建一个迷你 LLM 推理 infra。

学习目标按优先级：

1. LLM 推理引擎核心机制：模型前向、KV cache、prefill/decode 两阶段、采样
2. 自定义算子完整链路：HIP kernel → torch extension → `torch.ops` 调用 → 与原生算子对拍
3. （进阶）vLLM/SGLang 的核心：paged KV cache 与 continuous batching

首要目的是学习原理，不追求性能超越现有引擎，不追求产品化。

> 历史沿革：最初讨论过纯 C++ 五层全栈方案（rt/core/ops/engine/cli，GEMM 借 rocBLAS）。
> 经确认降级为"先吃透 PyTorch 层"路线；纯 C++ 栈保留为后续可选方向（§6）。

## 2. 环境（均已验证）

| 项 | 值 |
|---|---|
| GPU | AMD Radeon RX 9070 GRE（RDNA4，gfx1201，16GB），`/dev/kfd` 正常 |
| venv | `~/rocm10`，uv 创建，Python 3.14.7 |
| ROCm | 10.1 pip wheels（rocm_sdk_core / devel / libraries / device_gfx1201） |
| PyTorch | 2.14.0+rocm10.1.0，`torch.cuda` 可用，bf16 matmul 已跑通 |
| 工具链 | venv 内：hipcc、ninja；系统：cmake 4.4.4、ninja 1.13.2、g++ 16.2.1 |
| 其他包 | aiter 0.1.20、torchvision |
| 目标模型 | Qwen3-0.6B bf16（约 1.2GB，Apache-2.0 免授权；结构以 HF config.json 为准：28 层、hidden 1024、16 Q 头 / 8 KV 头、head_dim 128、QK-Norm、SwiGLU、tied embedding、RoPE、词表 151936） |

已知风险（由 P0 解决）：pip 版 ROCm 下 `torch.utils.cpp_extension` 编译需要调通环境变量
（`HIP_PATH`、`PYTORCH_ROCM_ARCH=gfx1201` 等）。这是第一个里程碑存在的意义。

## 3. 分阶段设计

### P0 — 打通自定义算子编译链路

- 用 `torch.utils.cpp_extension.load()` 现场编译第一个 HIP kernel（vector add），
  注册为 `torch.ops` 并在 Python 侧调用
- 把环境问题（hipcc 路径、gfx1201 架构声明）固化成一个可复用的加载脚本/环境说明
- **验收**：bf16 张量进出自定义 kernel 结果正确；重复执行不重编译（缓存生效）

### P1 — 纯 PyTorch 推理引擎（核心阶段）

不写任何 C++，不使用 transformers 的模型实现代码，全部手写：

- 权重加载：safetensors 直接读取（借 `safetensors` 库），按 HF config.json 构建参数表
- tokenizer：借 HF `tokenizers`（此层面分词器不是学习重点）
- Qwen3-0.6B 前向手写：RMSNorm、GQA attention（`torch.matmul`）、QK-Norm、
  SwiGLU、RoPE、tied embedding
- KV cache：contiguous 版（每层按 max_seq_len 预分配）
- 两阶段生成：prefill（整段）+ decode（逐 token）
- 采样：greedy → temperature → top-k → top-p 逐级实现
- 交互式 CLI（最简：stdin/stdout 循环即可）

**验收**：

1. 每个子模块（RMSNorm/RoPE/attention 块/整层）与 transformers 官方实现输出对拍一致
2. 固定 prompt + greedy 下，生成结果与 transformers **逐 token 一致**
3. 可交互对话

### P2 — 自定义算子替换

把 P1 引擎中的算子逐个换成手写 HIP kernel（走 P0 链路）：

1. RMSNorm → 自定义 kernel，与 `torch.rms_norm` 对拍 + benchmark
2. softmax（last dim）、RoPE、elementwise add/mul 同流程
3. 每个算子固定节奏：naive 版正确 → 对拍 → `torch.utils.benchmark` 测差距 →
   尝试一种优化（合并访问 / 向量化 / 减少 bank conflict）→ 复测

**验收**：至少 3 个算子完成"对拍一致 + 性能对比记录"；读懂一个 vLLM/SGLang
`csrc/` 中的真实 kernel（CUDA 或 HIP 版均可）并写一份阅读笔记（形式不限）。

### P3 — paged KV cache 与 continuous batching（进阶）

P1/P2 完成后再启动：

- paged KV cache 管理器：block table、槽位分配/释放、复用
- 手写 paged attention kernel（先 decode 路径，prefill 后续）
- continuous batching 调度循环：多请求并发 prefill/decode 混跑

**验收**：并发请求的输出与串行逐请求一致；吞吐相对 naive batching 有可测得的提升。

## 4. 借用与手写边界

| 类别 | 内容 |
|---|---|
| 借 | torch 全家（含 matmul/SDPA）、safetensors、HF tokenizers、`torch.utils.cpp_extension` |
| 手写 | 模型前向、KV cache、采样、生成循环、调度（P3）、全部自定义 kernel（P2/P3） |
| 不碰 | 训练/autograd、量化（awq/gguf/mxfp4）、GGUF 格式、分布式 |

## 5. 验证与错误处理

- **对拍（differential testing）是主要验证手段**：
  - P1：子模块与整模型对 transformers
  - P2：每个 kernel 对 torch 原生算子（bf16 容差 atol/rtol ~1e-2）
  - P3：并发输出对串行输出
- benchmark 统一用 `torch.utils.benchmark`
- 错误处理：shape/invariant 断言 fail-fast，带上下文（哪个模块、什么 shape）；
  不做静默回退；CLI 层负责把异常转成可读信息

## 6. 后续可选方向（不属于本项目范围）

- 纯 C++ 五层栈（原始方案 A：rt/core/ops/engine/cli）
- WMMA/MFMA 手写 GEMM、flash attention 完整版
- hiprtc 运行时编译、aiter 源码精读
- 量化推理路径
