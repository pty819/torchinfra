# FlyDSL gfx1201 工具链验证记录（2026-10-24，已打通）

目标：把 aiter 官方 FlyDSL bf16 flash attention（`flash_attn_func_gfx1201.py`）搬进
torchinfra，bench bf16 版 vs SDPA flash，作为 P14 fp8 路线的前置。**已全部完成。**

## 根因与最终方案

**"lld invocation failed" 的根因**：MLIR 的 HSACO 序列化（`gpu-module-to-binary`）
通过 `ROCM_PATH`/`ROCM_HOME` 定位 ld.lld 工具链；TheRock pip SDK
（`rocm_sdk_devel-10.1.0`）装进 venv 后**没有任何机制设置这些变量**，裸机上必然
失败——flydsl wheel、Debian mlir-opt 23、AMD TheRock mlir-opt 24 三套独立构建
全军覆没的原因全在此。AMD CI/容器里 ROCM_PATH=/opt/rocm 永远设着，所以只有
这种环境翻车。

**最终方案**（`flydsl_kernels/__init__.py` 导入时自动 setdefault，不碰全局）：

```
FLYDSL_GPU_ARCH=gfx1201                  # wheel 探测不到工具时默认回落 gfx942
ROCM_PATH=ROCM_HOME=$SITE/_rocm_sdk_devel
FLYDSL_COMPILE_LLVM_DIR=$SITE/_rocm_sdk_devel/lib/llvm   # AMD LLVM 24.0git
```

外部代码生成路径（子进程 mlir-opt → ld.lld，产 HSACO）与 flydsl bundled 的
MLIR 24.0git 同代同源（都是 ROCm/llvm-project fork），语法直通。

**注意**：flydsl wheel 自带的 in-process 序列化即使设好 ROCM_PATH 不报错了，
也会**静默产出错误结果**（kernel 未正确执行，Δ≈5）——外部路径必须保持默认。

## compat 四件套（flydsl_compat.py，不改 site-packages）

1. `flydsl.expr.buffer_ops` 属性注入（wheel 缺该模块，aiter kernel 要 import）；
2. `RocmBackend._format_pass_opts` 剥 `wave64`（wheel 的 Python 侧传此选项但自捆
   MLIR pass 不认识——wheel 内部组件不同步）；
3. generic-asm 桥接：bundled MLIR 24 的 pretty 语法（`fastmath<fast>` 属性）只有
   ≥24 认识；generic 形式打回老语法（`fastmathFlags = #llvm.fastmath<fast>`），
   对 23/24 都可解析。作用域仅限外部代码生成的文件写出；
4. work_dir 固定 `/tmp/flydsl_ext_debug`（留存中间文件便于排查）。
   另：主 kernel 的 `lsr-drop-solution` LLVM hint 在 wheel 上不认识，已删
   （仅调优语义）；`_run_compiled` cached 快路径 stream 需按位置传参。

## 基准（[1,32,S,128] bf16，Nemotron prefill 形状，同一口径 4·B·H·S²·D）

| S | flydsl | SDPA flash | 相对 | max\|Δ\| vs SDPA |
|---|---|---|---|---|
| 512 | 0.11ms / 40.2 TF | 0.08ms / 52.1 TF | 0.77x | 1e-4 |
| 2048 | 0.91ms / **75.8 TF** | 0.94ms / 72.9 | **1.04x** | 1e-4 |
| 8192 | 16.2ms / **67.8 TF** | 19.7ms / 55.8 | **1.21x** | 1e-4 |
| 32768 | 319ms / 55.1 TF | 322ms / 54.6 | 1.01x | 1e-4 |

调优扫描结论：默认 **BM=128/BN=32 即最优**（BM=64 慢 2x+，BN=64 慢 30%+，
waves_per_eu 仅 8K 处 wpe=1 比默认省 3%）。AMD 出厂已调好。

## 下一步

- **fp8 fork**：P14 的 fp8 数学（两遍式 rope+量化、P×448 自洽量化、分母同量化域）
  + 本文件的布局/流水手法（寄存器内 P 重打包、V 预取、WMMA 布局文档）= 互补两半。
  参照物：aiter PR #4188（open）。
- aiter main 要求的 `flydsl==0.2.1` pin 已过期，kernel 实际按 main API 写；
  0.4.1.dev993 + 本 compat 层可用。
- 清理：`~/ext/`（Debian 23 前缀 + debs，~200MB）是调试中间产物，可删；
  最终方案不依赖它。

## 附：fp8 WMMA 裸指令验证（2026-10-24 深夜，`tools/bench_flydsl_fp8_wmma.py`）

- **gfx1201 的 fp8 WMMA 形态是 `wmma_f32_16x16x16_fp8_fp8`（K=16，与 bf16 同代宽）**。
  x64/x128 变体是 gfx1250 专属——gfx1201 上会报 ISel "Cannot select intrinsic"。
- 碎片布局：A/B 每 lane **8B = v2i32**（fp8 按 dword 打包）。传 `vector<Nxi8>`
  会挂 ISel "Do not know how to split this operator's operand"。
- 兼容层新增：wheel 的 MLIR 打印 `modC = #rocdl.wmma_c_modifier<none>` 而
  TheRock 解析器不认该枚举拼写——属性可省略（默认即 none），compat 里已剥。
- **裸速率**（寄存器驻留操作数，独立累加链 NACC 掩盖延迟）：
  NACC=4 → 178 TF，NACC=8 → 220 TF，NACC=16 → **266.7 TF**
  （超过 P14 Triton 裸 dot 实测的 241 TF；NACC=16 已用 128 VGPR，近实用上限）。
- 结论：fp8 注意力 fork 的三块地基——指令形态、碎片布局、验证过的编译链——
  全部就位。bf16 flash 天花板 55–76 TF，fp8 裸速率 266 TF，**3.5–4.8x 计算余量
  真实可达**。

## 附2：fp8 flash attention 上线（P16，2026-10-24 深夜）

- 文件：`flydsl_kernels/flash_attn_fp8_gfx1201.py` + `fp8_wrapper.py`；
  `torchinfra/nemotron_rerank.py` 的 `_attn_fp8` 走 `FP8_ATTN_BACKEND='flydsl'`。
- 数学：P14 全套（descale 折 exp2 尺度、P×448、分母同量化域）+ 零同步
  device SCL 指针；GQA 用寻址数学（wave 不跨 query head）；
  真实 S/padded S 分离 → 任意 padding 长度安全（无 aiter 的 0.5% 限制）。
- 实测：GQA kernel 84–88 TF；e2e 32K 12.18→10.87s；E2E Δ=0.0403 持平基线。
- 调优结论：BN=32/wpe=1 定版（BN=64 VGPR 压力 -40%）；下一步 warp spec。

## 附3：V 转置布局与消融（GPT 协作轮，2026-10-24 夜）

- GPT 指路（PR#4188 的 V^T LDS 布局）验证有效但账要算清：fp8 下 GEMM2
  row-major 读 = 8×1B 标量读，转置后 = 1×8B 向量读，store 侧 +16×1B
  scatter——读写比 16:1 所以净赚（bf16 版 AMD 试过转置更慢是 1:1 不赚）。
- 正确实现净收益：8K 90.6→96.0 TF、32K 85.2→90.2；e2e 32K 10.87→10.49s。
- 两个坑：① 转置 tile LDS 尺寸按 D×stride（BN× 会越界，症状=精度掉
  18 倍 + 性能虚高 122 TF——假信号）；② V_T pad 必须 4（pad=2 的 34B
  stride 撞 bank 冲突掉到 33 TF，pad=0 也只有 65）。
- 消融（no_pv，DCE 会删 V 读，数字只看相对）：QK+softmax+store ≈1.8ms@8K，
  完整 9~11.5ms——**PV 段（V 读+P pack+GEMM2）占 ~80%**，是下一优化对象
  （合成 P 消融、PV 走 bf16、warp spec）。
- GPT 表格里"F lyDSL bf16 比 CK 慢 13%"是口径混用（旧 76 TF 数字）；
  同口径实测 CK=55.8@8K，flydsl bf16=67.8 反超 21%。

## 附4：GEMM2 fp8 原生布局设计分析（2026-10-24 夜，回答"能否设计 fp8 原生布局"）

**为什么转置躲不掉（硬件事实）**：WMMA 16x16x16 的 A/B 操作数布局是
"每 lane 8 个连续 K 维元素"（lane16=行/列，klane*8=K 偏移），A、B 同构。
PV 的收缩维是 S ⇒ V 无论当 A 还是 B，每 lane 都需要"固定 d 的 8 个连续
s 字节"；而 V 在显存里是 [s][d] 行主序。收缩维的布局要求是指令集固定的，
**转置必须发生在某一处**，能选的只是发生地点。

三个候选的指令账（每 lane 每 KV tile）：

| 方案 | V 侧指令 | barrier | 评估 |
|---|---|---|---|
| A. LDS scatter（现状，转置版） | 16×1B store + 16×8B 读 | 1 | 32 条 LDS |
| B. 寄存器转置（ds_bpermute 波内） | ~4 轮 permute × 8 个 dc 块 + byte repack ≈ 64 条 | 0 | 指令更多，依赖链更长——**不划算**（bpermute 本身也是 LDS 硬件） |
| C. 生产者转置（rope 写 V^T） | 0 转置，但 global 16B 粒度散射（~50% sector） | 1 | rope/attention 各付一次非合并 IO，约平手 |

实测消融已证明 V 侧总共只占 ~8-9%（no_pv 差值），所以 A→B/C 的上限
~5%；**80% 大头在 P pack 依赖链 + WMMA 发行 + softmax 标量链**。

本轮实验的两个 fp8 原生小改动（一成一退）：
1. **exp2 折叠 ×448 → 已回滚**：p·448 = exp2(x+log2(448)) 省 16 次乘，
   性能持平（VALU 有余量），但 log2(448) 是非整数常数，f32 加法的舍入
   是系统性的——kernel Δ 0.0013→0.0038、E2E 0.0403→0.1121（劣化 3 倍
   且逐层复利）。教训：**省乘法在 VALU 有余量时零收益，而精度门禁
   抓系统性噪声**；
2. **faithful_l 开关**：分母用 exp2 域直接求和（省 8 次 cvt 解码 +
   16 次加/tile）实测 +2.2~2.3% 性能、kernel Δ 0.002→0.006（仍远低于
   fp8 格点噪声）；默认保守保持 True，开关已留。

下一步真正的台阶（按预期收益排序）：
1. **跨 tile softmax 合并**：保持 BN=32 的 VGPR，两个 KV tile 一次
   m/l/corr 更新（corr 对 o_accs 的 8 次 v8f32 乘减半）——拿到 BN=64
   的部分收益而不付它的寄存器代价；
2. **跨 tile 流水**：GEMM1[t+1] 与 softmax/GEMM2[t] 重叠（消除
   barrier 串行），是 warp specialization 前的身位；
3. P pack 提前：S accumulator 出来即开始第一半 pack，与第二半
   softmax 标量链重叠。

## 附5：消融矩阵 → 三段串行结构；kv_merge/sw_pipeline 两实验一定案（2026-10-24 深夜）

### 消融矩阵（tools/bench_fp8_ablate.py，S=8192，总 11.7ms）

| 配置 | 跳过 | 实测 |
|---|---|---|
| none | — | 11.71ms |
| synth_p | P 构造链（exp2/fma/cvt/l），GEMM2 喂常数片段 | 3.86ms |
| no_pv | GEMM2+V（P 链保留） | 3.86ms |
| neither | 两者都跳（DCE 后只剩装载路径） | 3.86ms |

三个消融点**全部 3.86ms**、合起来 11.7ms ⇒ 内核是**三段纯串行**：装载段
（K/V global→LDS+scatter+barrier）≈3.85ms、exp2/pack 段≈3.9ms、GEMM2 段
≈3.9ms。任何一段单独存在都能隐藏在其他两段之下（synth_p/no_pv 与 neither
相等），合在一起逐段相加。**瓶颈是依赖串行，不是任何 pipe 的吞吐**——
VALU 只需 ~0.6ms 的量、wmma 也远未饱和（neither 段 GEMM1 等效 142 TF）。
neither 的 vgpr=20（DCE 掏空）说明"装载地板"其实是装载延迟本身。

### kv_merge=2（跨 tile softmax 合并）：正确但持平——已定位原因

实现完整（K/V 双 buffer、GEMM1 背靠背、softmax 一次合并、shuffle 3→1.5/ tile、
corr 校正减半），Δ 全过（含对抗 S=2050），性能持平（11.71→11.71）。原因：
GEMM1/exp2 段的 ILP×2 收益被 GEMM2 的 o_acc C-累加链×2 抵消（A、B tile
累加进同一组寄存器）。**教训：合并 ILP 只有在合并段彼此独立时才兑现**。
（merge2 的 V 双 buffer 还有波间 mod-2 竞态隐患，未修——反正不用了。）

### cheap_l 的 ×448 bug（历史记录错误）→ 修复后转正默认

cheap_l 分母用 exp2 域求和却漏乘 448（分子在 ×448 量化域），实测 Δ=6.24
（灾难级）；之前记录的"Δ 0.006"是错的。补 `_fmul(tile_sum, c_pmax)` 后
Δ=0.0007 与 faithful **完全持平**、+3.2~3.4% 性能（96.7/93.1 vs 93.9/90.5
TF @8K/32K）。**faithful_l 默认改为 False**。教训：量化域一致性
（分子分母同域）要有显式测试盯着。

### sw_pipeline：staged 干净但收益小；reorder 撞编译 bug

- **staged**（K/V 寄存器跨 barrier 暂存 + K/V 各 3 buffer LDS + 迭代开头单
  barrier）：数值与 legacy 逐位一致（3 buffer 因波间偏差 ≤1 迭代，mod 3 才
  无竞态），但仅 +1.3~1.6%，且与 cheap_l 收益重叠不叠加（96.8 → 95.1）——
  默认关闭，路径保留。
- **reorder**（barrier 挪到 pack 与 GEMM2 之间，GEMM2[t] 与 GEMM1[t+1] 间
  无 barrier）：**数值破坏**——Δ×10（S=512 达 0.0182，全行全头均匀 ~36%
  相对噪声），而 synth_p（常数 P）与 legacy 逐位相同 ⇒ 错误仅在"真 P 构造
  链 → GEMM2"路径。ISA 逐段核对（exp/fma/pack→B 操作数 v[84:87]、V 装载
  →A 操作数、l 链 fmac 累加）全部正确 ⇒ **疑似 TheRock mlir-opt/后端对
  "cvt_pk_fp8 半寄存器写（op_sel）产物跨 barrier 进 wmma 操作数"的错误
  编译**，未进一步定位。且 buggy 版也只有 -3.7%——即使修好，迭代内
  GEMM1→softmax→pack→GEMM2 依赖链仍主导。**下一步若再战流水，应做计算
  本身的跨迭代流水（S[t+1] 在 GEMM2[t] 之前算，需双 S 寄存器组），而不是
  装载流水**——那是 merge=2 + 独立 o_acc 双累加器（8 dc 各自独立链）的
  组合，代价 ~64 VGPR，接近 256 上限。

### FlyDSL 工程坑（本轮新发现，均已在代码里规避）

1. **yield 循环不能嵌在 if/else 分支里**：AST 改写器把含 loop-carried 状态
   的 if 拆成 __then/__else 函数时会丢外层绑定（NameError: loop_results）。
   对策：build 层定义两个 kernel def（_kernel_pipe/_kernel_legacy），launch
   按 SW_PIPE 选择。
2. **jit 函数里 kernel 调用必须直呼其名**：走别名/条件表达式会绕过改写器
   的调用点识别，得到裸 generator。对策：jit 函数整体在 build 层 if/else
   各定义一份。
3. 脚本拼接代码时 8 缩进 `else:` 锚点会撞公共段 causal else——本轮一次
   文件重组事故的根因，最终靠 AST yield-宿主映射校验修复。**教训：机械
   改大文件必须带 AST 级断言**。

### 定版数字（P16.1）

- kernel：8K **96.7 TF**（0.79/11.37ms）、32K **93.1 TF**（188.9ms），
  Δ vs dequant-ref 0.0013/0.0007/0.0004（S=2K/8K/32K）；
- e2e 门禁：8 对 CPU bf16 参考全过（max|Δ|=0.0914，排序一致 True），
  5 档延迟基准正常（8x64 14.3ms … 200x512 6177.8ms）；
- 默认：transposed V(pad4) + **cheap_l（×448 补偿）** + kv_merge=1 +
  sw_pipeline=False。
