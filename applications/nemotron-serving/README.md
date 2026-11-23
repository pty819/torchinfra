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
uv run nemotron-serving --host 0.0.0.0 --port 8000 \
  --model ~/models/Nemotron-3.5-Rerank-8B-FP8S
```

`--backend auto` 按模型 config 的 architectures 选择 Nemotron pointwise 或
Jina bi-encoder；原有 `/v1/rerank`、`/rerank`、`/health`、`/v1/models` 不变。

## 测试

```bash
cd applications/nemotron-serving
uv run pytest -m "not gpu"
uv run pytest -m gpu   # 需要模型权重和 AMD GPU
```
