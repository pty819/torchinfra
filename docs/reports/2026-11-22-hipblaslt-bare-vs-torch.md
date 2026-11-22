# hipBLASLt 裸调 vs torch._scaled_mm 对照实验

方法: 纯 C 垫片(SDK clang++ 编译, hipMalloc 自管理, heuristic 10 algo 全扫) vs torch API, 同形状同 GPU。
关键坑(踩了一路): hipBLASLt 的 hipblasOperation_t 是字符码 N=111/T=112; COMPUTE_32F=2(非 cuBLAS 的 68);
algo 参数按指针传(非 cuBLAS 的传值); 启发式需要 ≥80MB workspace 上限否则返回 0 个(只给 64MB 会只剩坏核);
B8B8S 家族 kernel 会异步摧毁 HIP context; 0x33 低翻转填充会因功耗/频率虚高性能 ~12%(数据相关降频伪影)。

## 结论(随机数据, 三形状)
| 形状 | torch | 裸调最优 |
|---|---|---|
| 7102×4096×12288 | 137.0 TF | 136.8 TF |
| 7102×4096×4096 | 139.0 TF | 140.4 TF |
| 7102×12288×4096 | 54.0 TF | 49.9 TF |

1. torch 调度层开销 = 0 (数字重合, 差 <1%)
2. torch 的 heuristic 选择 = 实测最优 (10 候选全扫确认)
3. 真实效率分层: gate/up & qkvo ≈ 51-53% 峰值; mlp_down (K=12288) 仅 ≈19-20% — reduction-heavy 形状是硬洼地, 全部 10 个 algo 都 ~50 TF, 非选择问题
4. 之前 profile 的 99 TF 平均被 down 形状拖低
