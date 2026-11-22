# qwen21dit

Qwen-Image-2.1 的手写 PyTorch 推理引擎 + 自研 ROCm/RDNA4 算子，运行于
AMD Radeon RX 9070 GRE (gfx1201)。不依赖 diffusers，直接加载 fp8 权重，
采样路径为自研 fp8 注意力内核 + hipBLASLt GEMM。

## 现状

- 1344×1344 · 20 步 er_sde · CFG 4 · 固定 LoRA 三件套：**常驻 ~97 s/图**
  （文本编码 ~1s + 采样 67-70s + VAE 解码 6-10s）
- 采样段速度 ≈ llama.cpp 系 (sdcpp) 的 **~2.2×**
- `serve.py` 常驻 worker（HTTP 7861）+ web 面板（预热 / 提示词 / 进度 / 历史）

## 用法

```bash
~/rocm10/bin/python phase_a/serve.py          # 常驻 worker :7861
python3 tools/qwen_web_app_minimal.py         # web 面板 :7860
```

## 目录

- `phase_a/` 引擎本体：dit_forward / sample / loader / encoder / serve / fp8_attn / fused_attn_input / lora
- `phase_a/tests/` 数值正确性测试
- `docs/` 设计 spec、实施计划、验收报告
- `tools/` 启动脚本与 web 面板

## 性能工作全史

> 记录所有尝试（含定案无效的），避免重复劳动。硬件背景：RDNA4 =
> fp8 WMMA 矩阵单元当代 + Ampere 级搬运能力（无 TMA/异步搬运引擎）。

### 有效的（按收益排序）

1. **自研 fp8 flash attention**（flydsl/WMMA）：92.6 TF @ MHA 32/32、
   S=7102 = 34.7% WMMA 峰值（266.7 TF）。替代 SDPA，采样提速 ~2.2×。
   fp8 全速率 WMMA 是 RDNA4 上唯一的大硬件红利。
2. **rope+quant 融合 Triton kernel**（B1+B3）：注意力输入单遍完成
   RoPE/拼接/量化，端到端 73→67s，attention bucket 510→275ms。
3. **常驻 worker + 预热编译**：消除 dynamo 每图 ~34s 重编译
   （含负向长度预热、cache_size_limit=128）；encoder 权重懒解压。
4. **er_sde 采样器**（sdcpp denoiser 忠实移植）：质量项，非速度。
5. **fp8 权重上挂 LoRA**：bf16 旁路不触碰 fp8 qdata。

### 定案无效的（负结果存档）

1. **warp-split 注意力**（wsplit → wsplit2）：按处方完整实现成对展开 +
   K/V 双缓冲 + 任意 S 掩码，正确性 Δ≤0.0018 全配置达标，但 **50 TF vs
   串行结构 92.6 TF（0.54×）**。根因：无 TMA 硬件下，wave 分组劈 KV 不
   创造计算资源，却使 K/V 搬运次数 ×2（q tile 64 vs 128 行），另付跨组
   merge + 双计数器开销。串行 8-wave 合搬摊销不可战胜。（torchinfra 仓库
   wsplit2 存档；历史 74 TF 读数系测量虚高。）
2. **hipBLASLt 裸调调研**（纯 C 垫片 + LD_PRELOAD，报告见 docs）：
   `torch._scaled_mm` 调度层零开销、启发式算法即实测最优 → **GEMM 层
   已在库天花板**，无自研空间。各形状：qkvo 139 TF（52% 峰值）、
   gate/up 137 TF、down 投影 ~50 TF（19%）——down 是 K 维归约硬件洼地，
   非软件可救。随机数据基准，0x33 填充会虚高 ~12%。
3. **B2 批量 CFG + key 掩码**：数学正确但 77s vs 基线 73s，未过 ≤70s 门
   （隔离在 phase-b2-attempt 分支）。附带教训：无掩码时零填充会毒害
   cond/unreg 分支，cond/neg 必须分开 forward。
4. **B3 2D grid 版本**：7168 micro-CTA 原子风暴 → 105s 回归，回退 1D。
5. **VAE tiling**：仅在显存不足时应急（解码显著变慢），默认关闭。
6. **encoder torch.compile**：46s 编译换 8-13%，不值；CPU eager 编码
   0.5-2.3s 更优，权重懒解压首 forward 9.4s 只付一次。

### 已识别未做

- **CUDA Graphs**：吞 launch 间隙，预计端到端 ~10s
- **少步蒸馏 LoRA**（Viggle 6-step 方向）：采样 67→~20s 的最大单项杠杆

### 硬件上限的结论

34.7% 注意力利用率是**结构级理性上限**：同机 hipBLASLt 裸 GEMM 也只有
52%，attention 定义内的 softmax 标量链/pack/转置约占剩余差额；指令级
残余预计可再 +10-20%，但属人肉 ISA 级投入且被端到端占比稀释。GEMM 与
attention 两层均已到达该硬件的合理水位，下一档收益在框架层（graph 捕获）
与算法层（蒸馏）。

## 许可

权重与 LoRA 遵循各自上游研究许可（非商用）。代码仅供学习研究。
