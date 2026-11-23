# qwen21dit

`torchinfra` 仓库中的 Qwen-Image-2.1 生成应用；算子由 `torchinfra` 提供。

Qwen-Image-2.1 的手写 PyTorch 推理引擎 + 自研 ROCm/RDNA4 算子，运行于
AMD Radeon RX 9070 GRE (gfx1201) / Ryzen 7 9700X。DiT 直接加载 fp8 权重，
采样路径为自研 fp8 注意力内核 + hipBLASLt GEMM，VAE 使用 Diffusers 实现，
文本编码器的线性层为 W8A8 int8（VNNI）。

## 现状（2026-11-23）

- 1344×1344 · 20 步 er_sde · CFG 4 · 固定 LoRA 三件套：**79.5 s/图**（热路径）
  （encode 0.5s + sample 72.8s + VAE 解码 5.6s）
- 采样段速度 ≈ llama.cpp 系 (sdcpp) 的 **~2.2×**
- 文本编码器 hidden cos **0.9984**（纯 W8A8，无混合精度）：sqrot2 = 全蝶形
  CPU 旋转（组内 B4^⊗4 + 组间 B4^⊗2/O3⊗B4^⊗2，torch.compile 融合）+ qkv/gateup
  组入口合并（旋转 252→144 次），编码零 GPU 依赖；enc 0.5s 与 GPU 版持平
  （该长度下瓶颈是 252 次 GEMM 派发而非旋转）。RSS 13.4→16.5GB 档
  （原 bf16 全量 21.4GB + 11.8GB swap → 系统可用 1.5→19GB）
- `qwen21dit.serve` 常驻 worker（HTTP 7861，含 dec_s/dec_gpu_s 分段计时、/selftest
  诊断端点）+ web 面板

## 用法

```bash
# 在 torchinfra 仓库根目录
uv sync --all-packages --all-extras
.venv/bin/rocm-sdk init       # 首次安装/重建 .venv 后必须执行
uv run qwen21dit-serve        # 默认: W8A8 蝶形合并版
# QDIT_ENC=sqrot uv run qwen21dit-serve  # 退回 GPU 旋转版 (长 prompt b≥500 时编码更快)
uv run qwen21dit-web         # web 面板 :7860，默认绑定 Tailscale IP / 0.0.0.0
```

ROCm devel payload 是惰性展开的；不执行 `rocm-sdk init` 时 JIT 会报
`lld invocation failed` 或 `thrust/complex.h file not found`。

## 目录

- `src/qwen21dit/`：运行时代码（worker、DiT、编码器、采样、Gradio 面板）。
- `scripts/`：离线校准、权重转换、benchmark 和 profiling。
- `tests/`：无 GPU 回归测试与算子单测。
- `tools/`：`qwen-draw.sh`、`qwen-web.sh` 启停脚本。

`QDIT_MODEL_DIR` 指定权重根目录（默认 `~/models/qwen21`）；`QDIT_OUTDIR` 指定出图目录
（默认 `~/Pictures/qwen`，worker 和网页应设为同一个目录）。自研 FP8 注意力由 workspace
依赖 `torchinfra` 提供，不需要再设置 `PYTHONPATH` 或同级目录。

### 适配的权重

下表的本地文件名均相对于 `QDIT_MODEL_DIR`。仓库不附带模型权重。

| 组件 | Hugging Face 仓库 / 文件 | 本地文件名 |
| --- | --- | --- |
| DiT · FP8 E4M3FN | [unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8/blob/main/Qwen-Image-2.1-FP8.safetensors) · `Qwen-Image-2.1-FP8.safetensors` | `fp8/qwen21_fp8_e4m3fn.safetensors` |
| VAE · Diffusers 权重 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/vae/diffusion_pytorch_model.safetensors) · `vae/diffusion_pytorch_model.safetensors` | `qwen_image_2.1_vae_diffusers.safetensors` |
| VAE · 配置 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/vae/config.json) · `vae/config.json` | `fp8/vae21_config.json` |
| 文本编码器 · Qwen3-VL-8B-Instruct FP8 | 上游模型参考：[Qwen/Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8)（两片 `model-*.safetensors`，另有模型配置和 tokenizer） | 离线工具使用 `fp8/qwen3vl8b_fp8/` 的单文件 `model.safetensors`、配置和 tokenizer；不是直接读取上游两片 |
| 文本编码器 · 运行时 W8A8 | **不是 Hugging Face 下载文件**。由项目离线校准和转换生成 | `int8_qwen3vl8b_sqrot2.safetensors` + `int8_qwen3vl8b_sq.safetensors`（补齐归一化等未量化张量），另需上述配置和 tokenizer |

DiT 和 VAE 的文件与上游 SHA-256 一致。测试用的编码器是本地单文件包，未确认它对应
Hugging Face 上的哪一个确切 revision；不能把上游分片直接改名替代，也不要用 Unsloth 的
ComfyUI 单文件 `Qwen-Image-2.1-text_encoder-FP8.safetensors` 替代这一输入格式。

默认还加载三份独立 LoRA，均放在 `loras/` 下：

- `nsfw_alpacas_v2.safetensors`：与 [Sentinel7/qwen-image-2.1 的 Alpacas V2](https://huggingface.co/Sentinel7/qwen-image-2.1/blob/main/2958918/3357315/NSFW%20Qwen%20by%20TheseAlpacas%20V2.safetensors) 文件哈希一致。
- `nsfw_genitalia_fix.safetensors`：与 [perpetual3x 的 `Genitalia-Anatomy-2.1-v1.safetensors`](https://huggingface.co/perpetual3x/Qwen-Image-Genitalia-Anatomy-Fix-NSFW-LoRA-v1/blob/main/Genitalia-Anatomy-2.1-v1.safetensors) 文件哈希一致。
- `nsfw_vaginus_v13b.safetensors`：测试使用本地 V1.3b 文件，未核验到对应的 Hugging Face 文件。

如不使用这组 LoRA，启动 worker 时设置 `QDIT_LORA=0`。实验用的 Omni 权重不是上表中的生产 DiT。

### 网页面板：卸载与刷新重连

- 默认在任务结束后空闲 **300 秒**自动卸载；也可点「卸载（释放内存和显存）」。
  有任务运行或排队时拒绝手动卸载；卸载过程中进来的生成请求会排队，完成卸载后再加载。
  在启动 worker 前设置 `QDIT_IDLE_S=0` 可关闭自动卸载。
- 状态每 5 秒刷新。冷启动的加载、预热与生成是同一个后台任务；刷新页面可重连最近一次生成，
  完成的 PNG 会出现在历史里。网页等待超时不会取消后台任务。worker 重启会丢失任务记录，
  但不会删除已保存的 PNG 或磁盘上的模型权重。
- RX 9070 GRE 实测：手动与自动卸载均将 PyTorch 已分配的显存从 **7.6 GiB** 降到 **0 GiB**；
  自动卸载在约 **302 秒**时确认完成，卸载后整卡显存占用约 **0.55 GiB**。
  卸载会释放常驻的模型权重，但不会退出 worker，Python、ROCm 和编译缓存仍可能占用系统内存。
- 无 GPU 回归测试：`cd applications/qwen21dit && uv run pytest -m "not gpu"`。

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
   - sqrot2 蝶形化（方案1+2）：两级旋转全部改 O(K·logK) 加减蝶形
     （compiled b=80 单次 0.27ms vs matmul 14.6ms），qkv/gateup 共享 smooth s
     后旋转入口合并 252→144；cos 0.9979→**0.9984**，编码摆脱 GPU/PCIe。
     诚实结论：典型 prompt (b≈40) 下 enc 墙钟不变（0.5s，瓶颈是 GEMM 派发）；
     b=506 时 CPU 蝶形与 GEMM 抢核反输 GPU 版 11%（7.3 vs 6.6s）。价值在
     cos 提升 + 编码与 GPU 采样可并行。权重: int8_qwen3vl8b_sqrot2.safetensors
5. **er_sde 采样器**（sdcpp 忠实移植）：质量项。
6. **fp8 权重上挂 LoRA**：bf16 旁路不触碰 fp8 qdata。
7. **DiT 动态编译（2026-11-23，8dfc9db）**：消除"每个新 prompt 长度首图
   +30~70s"重编译税。三件套缺一不可：`prefix_len` 从入参 int（dynamo 按值
   特化）改为模型内 `ctx.shape[1]` 推导；rope 构建移出编译区（`_pe` dict
   缓存 key 含 LT 也会特化）；`compile(dynamic=True)`。实测热路径 76.8s =
   静态 76.5s 零掉速，任意新长度 76.8-77.0s 免编译；裸 dynamic=True 不够
   （int 特化仍在，新长度照旧 153s）。代价：worker 重启后前 1-2 图各付 ~50s
   编译（inductor 磁盘缓存可摊薄）。零第三方库改动。

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
8. **int8 矩阵算力（为抄 SageAttention 做的硬件实测，2026-11-23）**：
   gfx1201 上 int8 与 fp8 **同速且 int8 略慢** —— 8192³ int8 `_int_mm`
   145.4 TOPS vs fp8 `_scaled_mm` 149.7 TF（**0.97×**）；8192×4096² 0.93×；
   4096³ 0.95×。内核名 `I8II` vs `F8BS` 同 tile MT128x128x32/MI16x16x1，
   证明走真张量核心而非模拟。旁证（NVIDIA 官方白皮书，int8 与 fp8 三代
   严格 1:1）：RTX 4090 FP8 660.6 = INT8 660.6；H100 SXM FP8 3958 = INT8
   3958；B200 FP8 4.5 = INT8 4.5 PFLOPS；B300 反而把 INT8 砍到 3（FP8 仍
   4.5）。"int8 高 2 倍"的三个真实来源：① INT4 vs INT8（位宽减半，4090
   1321 vs 660）；② 2:4 稀疏 vs 稠密（与格式无关）；③ FP8 vs FP16
   （SageAttention 吃的正是这步）。历史误解源于 A100 时代 INT8 2958 对比的
   是 FP16 312（FP8 当时不存在），跨代错配。另注：Ada 白皮书 FP8 660→330
   是 FP32 累加口径差异，与 RDNA4 现象同源——fp8 瓶颈常在累加器/输出转换。
   → 结论：QK 由 fp8 换 int8 无速度收益，此路线关闭。
9. **SageAttention（N 卡低精度注意力库）无可抄项**：kernel 为 CUDA 专属
   （Ampere/Hopper mma 形状、Sage3 用 Blackwell tcgen05+fp4），ROCm 无支持。
   算法层可抄的三项对我们是零或纯精度收益：① int8 QK（速度动机已被第 8 条
   证伪，只剩精度）；② Sage2 细粒度缩放（我们已是 tensorwise fp8，改
   per-block 仅精度保险，非必需）；③ smoothK 减均值（同上）。fp4 PV（Sage3）
   无硬件。其 2× 红利是"16 位→8 位"，我们早已吃掉（fp8 全链路）。
   → 与 warp-split 同档封存，避免重复调研。
   副产品：同机 fp8 GEMM 大 M 形状实测 **149-174 TF = bf16 的 1.6-1.7×**
   （8192³ 149.7、8192×4096² 173.8），高于此前记录的 hipBLASLt FP8 ~139 TF。

10. **2:4 结构化稀疏（稀疏算力）**：不仅是"我们用不上"，而是**业界已验证
   不值得**。硬件前提：权重侧 + 沿归约轴 + 固定 2:4 零 pattern，需稀疏化训练
   或 SparseGPT 类后训练剪枝恢复精度（vLLM 上唯一可用的例子重训了 130 亿
   token；一次性剪枝不可用）。实测天花板：指令级确有 2× 数学吞吐
   （NVIDIA arXiv 2104.08378），但端到端仅 **1.1~1.4×，L4 上有人测出 0×**
   （NVIDIA Model-Optimizer #1974 / TensorRT 论坛：sparse-eligible 层为 0）；
   attention/norm/访存不吃稀疏 + 稀疏 kernel 成熟度差 + 功耗墙，三重稀释。
   生态信号：**2026 年 vLLM 与 LLM Compressor 直接移除了 2:4 支持**（用的人
   太少）；SemiAnalysis：多数 AI 实验室生产推理忽略 2:4，转向量化与蒸馏。
   唯一成立场景是"已到硬件最低精度地板（FP8/NVFP4）且计算受限 + 有预算重训"
   ——H100 FP8+2:4 在 vLLM 上 prefill +27%（论文 2610.28947）。我们的位置：
   未到地板、非算力瓶颈（wall vs GPU <1s）、权重为官方稠密 fp8、ROCm 稀疏
   GEMM 生态只会更弱。→ 与 int8/fp4 同档排除，且性质更明确：不是"用不上"，
   是"行业验证过不值得"。

### 已识别未做

- **VAE dec 剩余 3.8s CPU dispatch 间隙**（与换页无关，worker 大图 20 万次
  dispatch 慢、小图正常之谜；子进程/compile 方案待做）
- **CUDA Graphs/HIP Graphs**（整段 kernel 录制+一键重放，消 CPU 发射开销）：
  11-20 时代估计 ~10s（当时 wall 68s vs GPU 57.7s 有 10.3s 间隙）；2026-11-23
  实测热路径 wall 76.8 vs GPU 76.5，采样段间隙已 <1s，剩余价值主要在 VAE
  dec 的 3.8s CPU dispatch 间隙。capture 需静态地址/形状，flydsl+hipBLASLt
  裸调需逐个验证兼容性
- **少步蒸馏 LoRA**（6-step）：67→~20s 的最大单项杠杆
- **GPTQ**：已评估——激活误差主导时（现格局）权重侧补偿无感，暂不做
- **"采样 2.06× 回归"系误报（2026-11-23 查明，非回归）**：逐新 prompt 测得
  149.7s 的真因是 **ctx 长度 LT 变化触发整图 dynamo 重编译（+29~70s/新长度）**，
  热路径实测 76.5s 与历史 72.8s 一致；fp8 GEMM 210TF/attention 266ms/裸 GEMM
  98TF 全部健康。该架构税（每个新 LT 首图慢 ~30-70s）已由"有效的"第 7 条
  dynamic 编译消除

### 环境坑记录

- 测试机的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 会直接 GPU core dump
- serve.py `busy=true` 是 queue 缺 task_done 的误报（显示问题）
- 新 prompt 长度触发 dynamo 重编译时 enc_s 首图会显示 ~10s（正常 0.5s）

## 许可

权重与 LoRA 遵循各自上游研究许可（非商用）。代码仅供学习研究。
