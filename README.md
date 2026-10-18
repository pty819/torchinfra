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
- [ ] P2 — 自定义算子替换（Triton：RMSNorm / softmax / RoPE）
- [ ] P3 — paged KV cache + continuous batching
