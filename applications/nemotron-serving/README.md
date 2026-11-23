# nemotron-serving

`torchinfra` 之上的 rerank HTTP 应用。模型实现（Jina/Nemotron）、调度、
HTTP API、批处理循环和进程入口都在本应用内；`torchinfra` 只提供 FP8
checkpoint、量化 GEMM、Triton 融合 kernel 等通用能力。

## 目录

- `src/nemotron_serving/jina_rerank.py`：Jina reranker 自研前向与融合 kernel。
- `src/nemotron_serving/nemotron_rerank.py`：Nemotron-3.5-Rerank-8B 自研前向与融合 kernel。
- `src/nemotron_serving/server.py`：`/v1/rerank` HTTP 服务、调度与进程入口。
- `scripts/`：rerank 基准、剖析和权重对拍脚本。
- `tests/gpu/`：需要本地权重与 GPU 的 pytest 对拍。

## 安装与运行

```bash
# 在 torchinfra 仓库根目录
uv sync --all-packages --all-extras
.venv/bin/rocm-sdk init       # 首次安装/重建 .venv 后必须执行
uv run nemotron-serving --host 0.0.0.0 --port 8000 \
  --model ~/models/Nemotron-3.5-Rerank-8B-FP8S
```

ROCm devel payload 是惰性展开的；不执行 `rocm-sdk init` 时 JIT 会报
`lld invocation failed` 或 `thrust/complex.h file not found`。

`--backend auto` 按模型 config 的 architectures 选择 Nemotron pointwise 或
Jina bi-encoder；原有 `/v1/rerank`、`/rerank`、`/health`、`/v1/models` 不变。

## 模型权重

仓库不附带权重。`--model` 需要指向包含 `config.json`、tokenizer、
`model.safetensors.index.json` 和分片的本地目录。

- **Jina**：上游 [jinaai/jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5)。
  BF16 目录可直接用；FP8 用 `tools/quantize_fp8_sharded.py` 生成：
  `uv run python tools/quantize_fp8_sharded.py ~/models/jina-reranker-v3.5 ~/models/jina-reranker-v3.5-fp8`
- **Nemotron**：上游 [nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16)。
  运行时推荐 FP8S（8.8GB），由 BF16 用 per-projection scale 转换：

  ```bash
  uv run python - <<'PY'
  from pathlib import Path
  from torchinfra.fp8_ckpt import save_fp8_checkpoint_sharded

  save_fp8_checkpoint_sharded(
      Path("~/models/Nemotron-3.5-Rerank-8B-BF16").expanduser(),
      Path("~/models/Nemotron-3.5-Rerank-8B-FP8S").expanduser(),
      split_scales=True,
  )
  PY
  ```

模型权重和 LoRA 遵循各自上游研究许可（部分非商用）。

## 许可

代码采用 [GPL-3.0-or-later](../../LICENSE)（见仓库根 `LICENSE`）。

## 测试

```bash
cd applications/nemotron-serving
uv run pytest -m "not gpu"
uv run pytest -m gpu   # 需要模型权重和 AMD GPU
```
