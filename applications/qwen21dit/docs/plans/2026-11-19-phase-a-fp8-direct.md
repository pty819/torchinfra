# Phase A 实施计划：PyTorch fp8 直跑（M0-M1 → Gate）

> **For agentic workers:** 用 subagent-driven-development 或 executing-plans 逐任务执行。步骤用 `- [ ]` 勾选。

**Goal:** unsloth fp8 底 + Omni LoRA 在测试机上以 PyTorch 常驻进程出图，四组口径测距，产出 Gate 报告（≤86s/张？）
**Architecture:** 手搭 Qwen-2.1 DiT 前向（PEFT 兼容命名）+ rocBLAS `_scaled_mm` fp8 GEMM + torch SDPA 注意力 + 自移植 sdcpp sigma 公式；diffusers 仅作正确性参考
**Tech Stack:** torch 2.14.0+rocm10.1（ROCm venv, py3.14）、transformers(Qwen3-VL ✓)、safetensors、现有 gradio 前端
**Spec:** docs/specs/2026-11-19-qwen21dit-fp8-engine-design.md

## Global Constraints
- 机器: 测试机 (RX 9070 GRE 12GB / Ryzen 9700X / 30GB RAM)，一切在 `qwen21dit/` 进行
- 权重: `~/models/qwen21/fp8/qwen21_fp8_e4m3fn.safetensors`(7.12GB) + `omni_lora_v0.1_bf16.safetensors`(0.67GB)（已下载中）
- 采样对齐: sdcpp `src/runtime/denoiser.hpp:736-810` FluxScheduler（qwen2.1: base_shift=0.5, max_shift=0.9, max_seq=8192, shift_terminal=0.02）
- 画质对拍基线: sdcpp 现网（Omni INT8 + alpacas@0.8, 🔞档）
- 提交纪律: 每任务一 commit，从仓库根目录运行 git

---

### Task 1: 权重勘察 → weights_report.json
**Files:** Create `phase_a/inspect_weights.py`, 产物 `phase_a/weights_report.json`
- [ ] 写脚本（完整可用）:
```python
import json, torch
from safetensors import safe_open
def inspect(path, out):
    r = {"path": path, "tensors": {}}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            t = f.get_slice(k)
            r["tensors"][k] = [list(t.get_shape()), str(t.get_dtype())]
    json.dump(r, open(out, "w"), indent=1)
from paths import MODEL_DIR

inspect(str(MODEL_DIR / "fp8/qwen21_fp8_e4m3fn.safetensors"), "phase_a/dit_report.json")
inspect(str(MODEL_DIR / "fp8/omni_lora_v0.1_bf16.safetensors"), "phase_a/omni_report.json")
inspect(str(MODEL_DIR / "loras/nsfw_alpacas_v2.safetensors"), "phase_a/alpacas_report.json")
```
- [ ] 运行 `python phase_a/inspect_weights.py`（前提: /tmp/qdit_dl 标记 DL_DONE 存在，否则等下载）
- [ ] 人工读三份 JSON，回答并记入 `phase_a/STRUCTURE.md`：①DiT 块数/hidden/头数（从 tensor 形状反推）②img_mlp 是 fused gate_up 还是 split（键名即答案）③fp8 dtype 是否带 scale 张量（无→需自标定）④Omni/alpacas LoRA 的键名约定与 rank
- [ ] Commit: `feat(a): 权重勘察与结构档案`

### Task 2: diffusers 支持度裁决
**Files:** Create `phase_a/probe_diffusers.py`，结论追加到 `phase_a/STRUCTURE.md`
- [ ] 脚本: `import diffusers; from diffusers import QwenImage21Pipeline`（下载装完后）; 若 ImportError 则尝试 `pip index` 查最新版本号并记录
- [ ] 判据（写死）: 能 import 且官方 `Qwen/Qwen-Image-2.1` diffusers 目录结构存在 → diffusers 路记为「参考实现」（用来对拍 forward 数学，不当运行时）；否则直接用 sdcpp 源码 + checkpoint 反推
- [ ] Commit（与 Task 1 可合并）

### Task 3: DiT 前向（PEFT 兼容命名）
**Files:** Create `phase_a/dit_forward.py`（本任务只搭 bf16 骨架+正确性，fp8 在 Task 4）
**Interfaces:** Produces `class QwenDiT(nn.Module)`，`forward(hidden, cond_vec, timesteps, rope) -> eps`；模块命名与 alpacas/omni LoRA 键名逐字一致（Task 1 ④ 的结论为准）
- [ ] 按 STRUCTURE.md 的块结构写 `__init__`：每块 = RMSNorm → img_attn(qkv+o) / txt_attn(若分离) → RMSNorm → img_mlp(gate_up+proj) + txt_mlp + AdaLN 调制（timestep-MLP 出 6/12 个调制系数乘加）
- [ ] 注意力 v0 用 `torch.nn.functional.scaled_dot_product_attention`（双向无 mask，text+img 拼接序列），RoPE 用 qwen mrope 的 3 频段实现（参考 diffusers QwenImage2TVModel 或 sdcpp 源 `model/qwen_image*.cpp` 的 rope 表）
- [ ] 单测 `tests/test_forward.py::test_shapes_and_finite`: 随机输入 forward → 形状回原样、无 NaN（CPU 上跑 2 块的小模型即可：允许 `num_blocks=2` 构造参数）
- [ ] Commit: `feat(a): DiT bf16 前向骨架`

### Task 4: fp8 GEMM 层
**Files:** Modify `phase_a/dit_forward.py`（linear 替换）；Create `phase_a/fp8_linear.py`
- [ ] `fp8_linear.py`（完整）:
```python
import torch
class FP8Linear(torch.nn.Module):
    """W 存 float8_e4m3fn + per-tensor scale；激活动态 per-tensor 量化；rocBLAS _scaled_mm"""
    def __init__(self, w_fp8: torch.Tensor, scale: float, bias=None):
        super().__init__()
        self.register_buffer("w", w_fp8)          # [out, in] e4m3fn
        self.register_buffer("ws", torch.tensor(scale))
        self.bias = bias
    def forward(self, x):
        xa = x.abs().amax().clamp(min=1e-6)
        xs = (xa / 448.0).float()
        xq = (x.float() / xs).clamp(-448, 448).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.w.t(), scale_a=xs, scale_b=self.ws,
                             bias=self.bias, out_dtype=torch.bfloat16)
        return y
def from_bf16(w_bf16):  # 离线标定
    s = (w_bf16.abs().amax() / 448.0).clamp(min=1e-12).float()
    return FP8Linear((w_bf16.float()/s).clamp(-448,448).to(torch.float8_e4m3fn), s)
```
- [ ] 单测 `tests/test_fp8_linear.py::test_vs_bf16`: 随机 [4096,4096] 阵，FP8Linear vs bf16 matmul 相对误差 < 3%（GPU 跑，`python -m pytest`）
- [ ] 加载器分支：Task 1 ③ 若 checkpoint 无 scale → 载入时按 `from_bf16` 同式自标定（fp8→float→amax/448）
- [ ] Commit: `feat(a): fp8 _scaled_mm 线性层`

### Task 5: LoRA 叠加器
**Files:** Create `phase_a/lora.py`
- [ ] 完整实现: 读 PEFT safetensors → `{module_name: (A[B float16], alpha)}`；`apply_all(dit, strength_map)`: 对每个命中模块包 `LoRAWrapper(base, A, B, s*alpha/r)`
```python
class LoRAWrapper(torch.nn.Module):
    def __init__(self, base, A, B, mult):
        super().__init__(); self.base, self.A, self.B, self.mult = base, A, B, mult
    def forward(self, x):
        out = self.base(x)
        if isinstance(out, torch.Tensor):
            return out + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.mult
        return out  # 非 tensor 返回（如 FP8Linear 未来内联）再议
```
- [ ] 单测: 用 omni_lora 真权重加载 → 命中张量数 == omni_report.json 键数（应 ≈384 或其 LoRA 版块数）；alpacas 同理
- [ ] Commit: `feat(a): PEFT LoRA 运行时叠加`

### Task 6: sigma 与采样循环
**Files:** Create `phase_a/sigmas.py`（移植 sdcpp 公式）+ `phase_a/sample.py`
- [ ] `sigmas.py`（完整，qwen2.1 常量写死）:
```python
import math, torch
BASE, MAX_SHIFT, MAX_SEQ, TERMINAL = 0.5, 0.9, 8192, 0.02
def compute_mu(seq_len):
    m = (MAX_SHIFT - BASE) / (MAX_SEQ - 4096)
    b = BASE - m * 4096
    return seq_len * m + b
def time_shift(mu, t):
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
def flux_sigmas(steps, seq_len):
    mu = compute_mu(seq_len)
    sig = [time_shift(mu, 1.0 - i / steps) if (1.0 - i / steps) > 0 else 0.0 for i in range(steps + 1)]
    if TERMINAL > 0 and steps > 1:  # 尾端重标（denoiser.hpp:792-810，余 4 行照源码抄）
        f = (1.0 - sig[steps - 1]) / (1.0 - TERMINAL)
        sig = [TERMINAL + (1.0 - s) * f for s in sig]; sig[steps] = 0.0
    return torch.tensor(sig)
```
- [ ] 单测 `tests/test_sigmas.py`: steps=20, seq_len=7056 → mu≈0.843、sigmas 单调降、首=1 尾=0（与 sdcpp 日志值对表：mu=0.843）
- [ ] `sample.py`: euler 与 er_sde 两种步进 + CFG 双批 concat（`torch.cat([cond,uncond])` 一次 forward 再拆）；sigma 缩放的模型包装（flow-matching: v-pred，x = x/sigma'... 按 sdcpp `stable-diffusion.cpp` 采样器段移植）
- [ ] Commit: `feat(a): 采样循环与 sigma 对齐`

### Task 7: 端到端冒烟 + 四组测距
**Files:** Create `phase_a/bench.py`；产物 `phase_a/BENCH.md`
- [ ] 冒烟: 512² "a lovely cat holding a sign says 'fp8'"，出图存 `phase_a/out/smoke.png`，断言非全黑/非 NaN（像素 std > 0.05）
- [ ] 四组: {1024², 1344²} × {Viggle 6步(CFG1), 🔞 20步(CFG4)}，各记 cold/warm、torch.cuda.synchronize 计时 + rocprof 分桶（GEMM/SDPA/other 三桶 kernel 时间）
- [ ] BENCH.md 表格 vs sdcpp 基线（111.6s@1344²/20步；Viggle 41s 采样@1344²）
- [ ] Commit: `feat(a): 端到端与测距`

### Task 8: 常驻服务 + gradio 接入
**Files:** Create `phase_a/serve.py`；Modify Gradio app（加"引擎"单选: sdcpp/qdit）
- [ ] serve.py: 起进程加载权重常驻，stdin/本地 socket 收 JSON(prompt, w, h, steps, cfg, sampler, seed, loras) → 出 PNG 路径；队列 1 并发
- [ ] app.py: 引擎= qdit 时 subprocess 调 serve（沿用现有 gr.Progress 流程，`bash -c source rocm-env` 不再需要）
- [ ] 网页实测一张 1344² 🔞，与 sdcpp 同参数出图对比
- [ ] Commit: `feat(a): 常驻服务接入网页`

### Task 9: Gate 报告
**Files:** Create `docs/reports/2026-11-19-gate.md`
- [ ] 数据: warm 1344²·20步·CFG4 采样秒数；与 86s 线比较；rocprof 瓶颈桶
- [ ] 结论二选一: A 达标（Phase B 缓行）/ B 未达标（列 Phase B 补差清单与预期收益）
- [ ] Commit + 通知用户决策

## Self-Review 结论
- 覆盖 spec §3 Phase A 全部；Gate/报告=Task 9；风险①②由 Task 1/2 裁决；风险③由 Task 7 冒烟暴露
- 无 TBD；Task 3 的形状常量来源=Task 1 产物（数据驱动非占位）；Task 6 尾端 4 行注明照抄源码行号
