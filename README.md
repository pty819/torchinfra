# qwen21dit

Qwen-Image-2.1 的手写 PyTorch 推理引擎 + 自研 ROCm/RDNA4 算子，运行于
AMD Radeon RX 9070 GRE (gfx1201) / Ryzen 7 9700X。不依赖 diffusers，
直接加载 fp8 权重，采样路径为自研 fp8 注意力内核 + hipBLASLt GEMM，
文本编码器为纯 W8A8 int8（VNNI）。

## 现状（2026-11-23）

- 1344×1344 · 20 步 er_sde · CFG 4 · 固定 LoRA 三件套：**79.5 s/图**（热路径）
  （encode 0.5s + sample 72.8s + VAE 解码 5.6s）
- 采样段速度 ≈ llama.cpp 系 (sdcpp) 的 **~2.2×**
- 文本编码器 hidden cos **0.9979**（纯 W8A8，无混合精度），内存 RSS 13.4GB
  （原 bf16 全量 21.4GB + 11.8GB swap → 系统可用 1.5→19GB）
- `serve.py` 常驻 worker（HTTP 7861，含 dec_s/dec_gpu_s 分段计时、/selftest
  诊断端点）+ web 面板

## 用法

```bash
QDIT_ENC=sqrot ~/rocm10/bin/python phase_a/serve.py   # 默认: W8A8 组合版
python3 tools/qwen_web_app_minimal.py                 # web 面板 :7860
```

## 性能工作全史

> 记录所有尝试（含定案无效的），避免重复劳动。硬件：RDNA4 = fp8 WMMA
> 当代 + Ampere 级搬运（无 TMA）；Zen5 = AVX512-VNNI 全速 int8。

### 有效的（按收益排序）

1. **自研 fp8 flash attention**（flydsl/WMMA）：92.6 TF @ MHA 32/32、
   S=7102 = 34.7% WMMA 峰值。替代 SDPA，采样提速 ~2.2×。
2. **rope+quant 融合 Triton kernel**（B1+B3）：73→67s，attention bucket
   510→275ms。
3. **常驻 worker + 预热编译**：消除 dynamo 每图 ~34s 重编译。
4. **encoder 纯 W8A8 三级演进**：
   - SmoothQuant(α=0.75) 校准迁移：hidden cos 0.988，encode 1.0→0.4s，
     权重落盘 9.87GB，RSS 21.4→13.4GB
   - × 两级随机旋转（组间 QR 正交 + 组内 256 阶正则 Hadamard）：
     cos 0.988→**0.9979**（smooth 迁移兜 massive activation，旋转管分布）
   - 旋转挪 GPU 执行：encode 20.1→**0.5s**（40×；CPU 小 matmul 被 16 线程
     fork/join 拖到 8 GFLOP/s 是根因）
5. **er_sde 采样器**（sdcpp 忠实移植）：质量项。
6. **fp8 权重上挂 LoRA**：bf16 旁路不触碰 fp8 qdata。

### 定案无效的（负结果存档）

1. **warp-split 注意力**：完整成对展开+双缓冲+任意 S 掩码，正确性达标
   但 50 TF vs 串行 92.6 TF（0.54×）。无 TMA 下 wave 分组必输 2× K/V 搬运
   账。历史 74 TF 读数系测量虚高。
2. **hipBLASLt 裸调**（纯 C 垫片）：torch 调度层零开销、heuristic 即最优
   → GEMM 已在库天花板。down 投影 ~50 TF（19%）是归约硬洼地。
3. **裸 W8A8 动态量化**（per-token + reduce_range）：cos 0.697 崩盘。
   Qwen3 down_proj 输入 massive activation 达 11000×（max 6144 vs
   median 0.54，layers 6/16/35）。
4. **纯旋转（无 smooth）**：cos 0.943。前 17 层逐层 0.9998+ 证明旋转对
   稀疏尖峰有效，但 L16/35 的 massive 旋转不散（DFRot 论文边界），
   终点离散跳变。
5. **混合降级层**（93 层 W8A16 兜底）：仅 +0.004 且逐层反量化慢，弃。
6. **B2 批量 CFG**、**B3 2D grid**、**VAE tiling**、**encoder compile**：
   各自未过门或负收益，详见 docs。
7. **CPU W4**：VNNI 无 int4 路径，eager 解包反而更慢；留待手写融合 kernel。

### 已识别未做

- **VAE dec 剩余 3.8s CPU dispatch 间隙**（与换页无关，worker 大图 20 万次
  dispatch 慢、小图正常之谜；子进程/compile 方案待做）
- **CUDA Graphs**：预计端到端 ~10s
- **少步蒸馏 LoRA**（6-step）：67→~20s 的最大单项杠杆
- **GPTQ**：已评估——激活误差主导时（现格局）权重侧补偿无感，暂不做

### 环境坑记录

- 82 的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 会直接 GPU core dump
- serve.py `busy=true` 是 queue 缺 task_done 的误报（显示问题）
- 新 prompt 长度触发 dynamo 重编译时 enc_s 首图会显示 ~10s（正常 0.5s）

## 许可

权重与 LoRA 遵循各自上游研究许可（非商用）。代码仅供学习研究。
