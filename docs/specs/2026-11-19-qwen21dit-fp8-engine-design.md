# qwen21dit — Qwen-Image-2.1 FP8 混合引擎设计（路线 C）

日期: 2026-11-19 · 状态: 待评审
基线: sdcpp(fp16 WMMA) 1344²·20步·er_sde·CFG4 采样 111.6s，全流程 ~160s（含每张冷启动 25s）

## 1. 目标与非目标

**目标**: 1344²·20步·CFG4 采样 ≤86s（1.3x）；总墙钟（常驻后）冲 60s 段；画质对拍与 sdcpp 偏差可接受。
**非目标**: 通用 DiT、LLM 统栈、量化研究、VAE/文本编码器优化。

## 2. 固定不变的层

- NSFW LoRA: alpacas v2 @0.8（+vaginus v1.3b / genitalia fix 可选叠加）— 与引擎无关，照常叠加
- Uncensored: unsloth/Qwen-Image-2.1-FP8（原生 fp8 底）+ chatrlabs-omni-lora-v0.1-bf16（运行时叠加）
- VAE: qwen_image_2.1_vae_bf16（现有文件）· 文本编码器: Qwen3-VL-8B（unsloth 有 fp8 版，编码一次可按 prompt 缓存）
- 采样配方: 沿用现网（euler/er_sde + 我们的 sigma 表 + 🔞/Viggle/Lightning 档位）

## 3. 路线 C：两阶段 + 数据闸门

### Phase A — PyTorch fp8 直跑（先做，工作量: 天级）
- 环境: torchinfra 的 ~/rocm10 venv（rocBLAS _scaled_mm fp8 实测 107~158 TF）
- 前向: 优先 diffusers 原生 QwenImage21 pipeline（M0 第一项: 验证 diffusers 支持度与 fp8 加载路径）；不通则手搭 DiT module 直读 unsloth safetensors
- 结构: 权重常驻（进程不退出，杀掉冷启动 25s）+ 采样循环 + CFG 双前向批 + prompt→embedding 缓存
- 测距口径: 同 sdcpp 完全一致的四组（1344²/1024² × 20步/4步），rocprof 分桶（GEMM/attn/胶水）

### Gate（数据闸门）
- warm 采样 ≤86s → Phase B 缓行，A 直接交付（接 gradio）
- warm 采样 >86s → Phase B 启动，B 的补差目标 = 86s 以下

### Phase B — 自建引擎增量（视 Gate，工作量: 周级）
- GEMM: rocBLAS _scaled_mm（不自研）
- Attention: FlyDSL FP8 WMMA（P16 骨架改 DiT 头布局 + CFG 批；Triton 55TF 兜底）
  - 前置: HSACO 外部链路实战（FLYDSL_COMPILE_LLVM_DIR + 官方 LLVM tarball + 系统 lld）
- 胶水: Triton 融合（RMSNorm+残差 / AdaLN / mRoPE）
- 移植核心资产: P×448 寄存器量化自洽、零同步 descale、真实S/paddedS 分离（头几何重写）

## 4. 目录（82）

```
~/项目/qwen21dit/
├─ docs/specs/          # 本文档
├─ phase_a/             # PyTorch fp8 直跑（M0 测距 + 常驻管线）
│   ├─ dit_forward.py   # diffusers 或手搭
│   ├─ loader_fp8.py    # unsloth fp8 + Omni LoRA 叠加
│   ├─ bench.py         # 四组测距 + rocprof 分桶
│   └─ serve.py         # 常驻进程，接现有 gradio（替换 sd-cli 起进程）
├─ phase_b/             # Gate 通过后启动
│   ├─ glue_triton/
│   └─ attn_flydsl/
└─ weights/             # 下载与 LoRA 离线 merge 脚本（可选优化）
```

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| diffusers 不支持 2.1 / fp8 加载怪癖（gfx1201 有先例：int8_convrot 黑图，换 fp8_scaled 修复） | M0 第一天验证；fallback 手搭前向 |
| unsloth fp8 无 scale 张量（scale=1 直转） | 加载时自标定 per-tensor scale |
| 12GB 显存（fp8 权重 ~7GB + 激活） | 11k 序列 < torchinfra 32K@10GiB 先例；必要时 MLP 序列分块 |
| FlyDSL HSACO 链路未实战 | M4 首项打穿；Triton attention 兜底保 1.3x |
| 对拍漂移过大 | 固定 5 prompt × 3 seed 集，LPIPS + 人眼，阈值验收时定 |

## 6. 里程碑与退出标准

- M0: 环境+diffusers 验证+基线四组测距 → 产出速度对照表
- M1: Phase A 常驻管线跑通 → warm 端到端出图，gradio 可切
- **Gate 评审（本 spec 的核心决策点）**
- M2(若启): HSACO 链路 → Triton 胶水 → FlyDSL attention → 引擎总装
- 收尾: 精度回归 + 稳定性 24h + 网页切换开关
