# nemotron-serving

`torchinfra` 之上的 rerank HTTP 应用。模型加载和调度仍在 `torchinfra`；
这里拥有 HTTP API、批处理循环和进程入口。

```bash
# 在 torchinfra 仓库根目录
uv sync --all-packages
uv run nemotron-serving --host 0.0.0.0 --port 8000 \
  --model ~/models/Nemotron-3.5-Rerank-8B-FP8S
```

`--backend auto` 按模型 config 的 architectures 选择 Nemotron pointwise 或
Jina bi-encoder；原有 `/v1/rerank`、`/rerank`、`/health`、`/v1/models` 不变。
