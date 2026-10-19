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

