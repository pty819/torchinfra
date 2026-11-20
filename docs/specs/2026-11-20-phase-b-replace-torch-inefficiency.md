# Phase B 规范：用自研 kernel 替换 torch 低效环节

日期: 2026-11-20 · 依据: profile_step.py 实测（1344², 7079 token, compiled, 每前向 1614ms）

## 基线（不可回归的锚点）
- 采样 73s（40 前向 ×1614ms + 8.4s 空隙）; 全流程 105s
- 口径: 1344² / 20步 / er_sde / CFG4 / seed 42 / compile 热缓存 / 3 次取中位
- 质量门: 同 seed 出图 z-vision 判定"猫+木牌+构图连贯"，且与基线图 PSNR 不低于阈值(B2: 数值重排类改动 ≥ 0.90 结构相似即可，接受微小色差)

## 时间分布与靶单
| 环节 | 占比 | 决策 |
|---|---|---|
| fp8 GEMM (hipBLASLt) | 62% | 不动（Triton 打不过，实测 99TF/s） |
| attention (flyc bf16) | 31.6% | B1: 自研 fp8 flash attention（FlyDSL P16 路线） |
| CFG 双前向串行 | 结构性 | B2: 恢复双批（pad + 掩码），前提=掩码不跌出 flash 路径 |
| 胶水 6% + 空隙 12% | 小 | B3: rope/norm/量化融合（最后做） |

## B2: CFG 双批恢复（先行，纯 torch）
- 前置实验: SDPA 加 [B,1,1,S_k] key-padding 掩码，ROCm 上是否仍走 flyc flash kernel
  - 若跌出 flash 路径 → B2 独立项取消，掩码能力并入 B1 自研 attention（kv_len per row 原生支持）
- 实现: DiT forward 接受 text_pad_len; neg 用零填充至 cond 长度; 掩码仅作用于 key 列
- 验收: 采样 ≤70s（−3s）且质量门通过; 失败即 git revert

## B1: fp8 flash attention（主攻）
- 路线: 复用 flydsl-fp8-flash-attention-p16 技能; QK-RMSNorm 后量化 e4m3; V 同
- 风险登记: HSACO 工具链（见 flydsl-hsaco-blocker 技能）需先探路; 数值对拍失败则回退 bf16 attention
- 验收: profile 复测 attention 桶 ≥ 减半; 采样 ≤63s; 质量门通过

## B3: epilogue 融合（顺手）
- qk-norm + rope + 量化三连融合; 目标胶水 6%→3%

## 纪律
- 每阶段独立 commit, 失败立即 revert, 不留实验代码在主干
- 机器保护: 编译类操作 TORCHINDUCTOR_COMPILE_THREADS=2; 内存看门狗 free≥3GB 才动手
- 禁止动 GEMM 路径与采样器语义
