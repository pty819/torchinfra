# flydsl_kernels

gfx1201 的 FlyDSL kernel。生产路径只有 FP8 attention：

| 文件 | 状态 | 用途 |
| --- | --- | --- |
| `flash_attn_fp8_gfx1201.py` | 生产 | 当前 FP8 FA（MHA/GQA）；qwen21dit 和 nemotron-serving 都走这里 |
| `fp8_wrapper.py` | 生产 | scale/pad/缓存包装；nemotron 的 `_attn_fp8` 入口 |
| `flydsl_compat.py`、`_buffer_ops_impl.py` | 生产兼容层 | FlyDSL 0.4.1 的 `buffer_ops` 注入 + MLIR 23/24 文本桥 |
| `kernels_common.py`、`tensor_shim.py` | vendored helpers | 上述 kernel 的公共辅助 |
| `flash_attn_func_gfx1201.py` | 参考基线 | aiter 的 BF16 FA，FP8 版本的移植骨架；生产不调用 |
| `wrapper.py` | 参考基线 | BF16 wrapper，仅 benchmark 用 |
| `flash_attn_fp8_wsplit.py` | 实验归档 | warp-split v1；正确但慢于生产（74 TF @8K/32K vs 99.7/93.4） |
| `flash_attn_fp8_wsplit2.py` | 实验归档 | warp-split v2（双缓冲/任意 S）；最终约 50 TF vs 生产 92.6 TF（0.54x），设计墙，未接入生产 |

生产配置是 legacy + cheap_l（`faithful_l=False`、`kv_merge=1`、
`sw_pipeline=False`），实测 100.7 TF @8K、93.9 TF @32K。warp-split 的负结果
和配方保留在 `README.md` P16.3 与 `docs/flydsl_gfx1201_toolchain.md` 附6。
