# P1 纯 PyTorch 推理引擎 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 不借助 transformers 的模型代码，纯 PyTorch 手写 Qwen3-0.6B 前向 + contiguous KV cache + prefill/decode 两阶段 + 采样 + 交互 chat，与 transformers 对拍一致。

**Architecture:** 模块按解析顺序自底向上构建：config/weights → RMSNorm/RoPE/KVCache → Attention → MLP/Layer/整模型 → tokenizer/sampling → 生成循环与 CLI。参数命名与 HF safetensors 完全一致，用 `load_state_dict` 的校验能力保证权重映射零遗漏。对拍策略：子模块用测试内独立实现的数学参考（不依赖 HF 内部 API，版本稳）；整模型用 HF 公开 API 的 logits/greedy 输出做金标准；验收测试双侧 fp32 消除 bf16 噪声。

**Tech Stack:** Python 3.14（`~/rocm10/bin/python`）、PyTorch 2.14+rocm10.1（bf16 on 9070 GRE）、safetensors、transformers（仅作对拍参考与 tokenizer）。

**Spec:** `docs/superpowers/specs/2026-10-19-torchinfra-design.md` §3 P1

## Global Constraints

- 所有 python 命令一律 `~/rocm10/bin/python`，在仓库根目录以 `PYTHONPATH=.` 运行
- 测试是纯脚本（无 pytest）：`tests/test_*.py` 内 `main()` + `assert`，失败即异常退出
- 引擎本体 bf16；对拍验收测试（Task 10）双侧用 fp32 消除浮点噪声
- 模块参数名必须与 HF 权重名逐字一致（`model.layers.0.self_attn.q_proj.weight` 等）
- 模型目录固定 `~/models/Qwen3-0.6B`
- 每个 Task 完成即 commit（沿用仓库既有 git 身份配置）

## 文件结构

```
torchinfra/
  config.py       # Qwen3Config dataclass + from_json
  weights.py      # safetensors → GPU bf16 权重 dict（含 tied lm_head 补齐）
  modules.py      # RMSNorm / RotaryEmbedding / Attention / MLP / DecoderLayer
  kv_cache.py     # KVCache（contiguous，单请求）
  model.py        # Qwen3Model / Qwen3ForCausalLM（HF 参数名 + from_pretrained）
  tokenizer.py    # TokenizerBox（借 HF AutoTokenizer）
  sampling.py     # sample_token（greedy/temperature/top-k/top-p）
  generate.py     # generate()（prefill+decode 流式生成）
  chat.py         # CLI 入口（python -m torchinfra.chat）
tests/
  _ref.py         # 共享：HF 参考模型懒加载、固定 prompt、公共工具
  test_config.py test_weights.py test_modules.py test_kv_cache.py
  test_model.py test_tokenizer.py test_sampling.py test_p1_acceptance.py
```

---

### Task 1: 模型下载与 config 解析

**Files:**
- Create: `torchinfra/config.py`
- Create: `torchinfra/__init__.py`（已存在则不动）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `@dataclass Qwen3Config`，字段 `vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, rope_theta, max_position_embeddings, tie_word_embeddings`；`Qwen3Config.from_json(path_or_dir: str | Path) -> Qwen3Config`（接受模型目录或 config.json 路径）

- [x] **Step 1: 下载 Qwen3-0.6B（约 1.2GB）**

```bash
~/rocm10/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-0.6B",
                      local_dir="~/models/Qwen3-0.6B")
print("downloaded to", p)
EOF
```

验证：`ls ~/models/Qwen3-0.6B` 含 `config.json` 与 `model.safetensors`。

- [x] **Step 2: 写失败测试**

```python
# tests/test_config.py
import json
from pathlib import Path
from torchinfra.config import Qwen3Config

MODEL_DIR = Path.home() / "models" / "Qwen3-0.6B"

def main():
    cfg = Qwen3Config.from_json(MODEL_DIR)
    raw = json.loads((MODEL_DIR / "config.json").read_text())
    assert cfg.hidden_size == raw["hidden_size"]
    assert cfg.num_attention_heads == raw["num_attention_heads"]
    assert cfg.head_dim == raw["head_dim"]
    assert cfg.tie_word_embeddings == raw.get("tie_word_embeddings", False)
    assert cfg.num_key_value_heads <= cfg.num_attention_heads
    assert cfg.head_dim % 2 == 0  # RoPE 需要
    print("config OK:", cfg)

if __name__ == "__main__":
    main()
```

- [x] **Step 3: 跑测试确认失败**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_config.py`
Expected: `ModuleNotFoundError: No module named 'torchinfra.config'`

- [x] **Step 4: 实现 config.py**

```python
# torchinfra/config.py
"""Qwen3 配置：直接映射 HF config.json 的字段。"""
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, path_or_dir: str | Path) -> "Qwen3Config":
        p = Path(path_or_dir)
        if p.is_dir():
            p = p / "config.json"
        raw = json.loads(p.read_text())
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", 10000.0),
            max_position_embeddings=raw.get("max_position_embeddings", 4096),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
        )
```

- [x] **Step 5: 跑测试通过 + commit**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_config.py` → `config OK: ...`

```bash
git add torchinfra/config.py tests/test_config.py
git commit -m "P1: 模型下载与 Qwen3Config 解析"
```

---

### Task 2: 权重加载（safetensors）

**Files:**
- Create: `torchinfra/weights.py`
- Test: `tests/test_weights.py`

**Interfaces:**
- Consumes: `Qwen3Config`（Task 1）
- Produces: `load_weights(model_dir: str | Path, device: str = "cuda", dtype = torch.bfloat16) -> dict[str, torch.Tensor]`；tied 模型缺 `lm_head.weight` 时自动补别名（与 embed 同一 tensor 对象）

- [x] **Step 1: 写失败测试**

```python
# tests/test_weights.py
from pathlib import Path
import torch
from torchinfra.config import Qwen3Config
from torchinfra.weights import load_weights

MODEL_DIR = Path.home() / "models" / "Qwen3-0.6B"

def main():
    cfg = Qwen3Config.from_json(MODEL_DIR)
    sd = load_weights(MODEL_DIR)
    assert sd["model.embed_tokens.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
    assert sd["model.embed_tokens.weight"].dtype == torch.bfloat16
    assert sd["model.embed_tokens.weight"].device.type == "cuda"
    # 每层必备权重名齐全
    for i in range(cfg.num_hidden_layers):
        for name in (f"model.layers.{i}.self_attn.q_proj.weight",
                     f"model.layers.{i}.self_attn.q_norm.weight",
                     f"model.layers.{i}.mlp.gate_proj.weight",
                     f"model.layers.{i}.input_layernorm.weight"):
            assert name in sd, f"缺少 {name}"
    assert "model.norm.weight" in sd
    # tied：loader 补出的 lm_head 与 embed 是同一个对象
    assert sd["lm_head.weight"] is sd["model.embed_tokens.weight"]
    print(f"weights OK: {len(sd)} 个张量")

if __name__ == "__main__":
    main()
```

- [x] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_weights.py`
Expected: `ModuleNotFoundError: No module named 'torchinfra.weights'`

- [x] **Step 3: 实现 weights.py**

```python
# torchinfra/weights.py
"""safetensors 权重加载：搬上 GPU、统一 dtype、补 tied lm_head。"""
from pathlib import Path
import torch
from safetensors.torch import load_file


def load_weights(model_dir: str | Path, device: str = "cuda",
                 dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    sd: dict[str, torch.Tensor] = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    if not sd:
        raise FileNotFoundError(f"{model_dir} 下没有 safetensors 文件")
    sd = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
    if "lm_head.weight" not in sd:
        if "model.embed_tokens.weight" not in sd:
            raise KeyError("既没有 lm_head.weight 也没有 embed_tokens，无法 tie")
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    return sd
```

- [x] **Step 4: 跑测试通过 + commit**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_weights.py` → `weights OK: ...`

```bash
git add torchinfra/weights.py tests/test_weights.py
git commit -m "P1: safetensors 权重加载（含 tied lm_head 补齐）"
```

---

### Task 3: RMSNorm

**Files:**
- Create: `torchinfra/modules.py`（本任务只有 RMSNorm，后续任务往里加）
- Test: `tests/test_modules.py`（文件后续任务共用，逐步追加测试函数）

**Interfaces:**
- Produces: `class RMSNorm(nn.Module)`：`__init__(dim: int, eps: float = 1e-6)`，参数名 `weight`；forward 输入输出同 shape 同 dtype，内部 fp32 计算

- [x] **Step 1: 写失败测试（测试内自带 fp32 独立参考实现）**

```python
# tests/test_modules.py — 本文件随任务追加
import torch
from torchinfra.config import Qwen3Config
from torchinfra.modules import RMSNorm


def test_rms_norm():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(2, 17, 64, device="cuda", dtype=dtype)
        m = RMSNorm(64, eps=1e-6).cuda().to(dtype)
        with torch.no_grad():
            m.weight.uniform_(0.5, 1.5)
        # 独立参考：全 fp32 手写公式
        x32 = x.float()
        ref = m.weight.float() * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6))
        out = m(x)
        torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
        assert out.dtype == dtype
    print("RMSNorm OK")


def main():
    test_rms_norm()

if __name__ == "__main__":
    main()
```

- [x] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_modules.py`
Expected: `ImportError: cannot import name 'RMSNorm'`

- [x] **Step 3: 实现**

```python
# torchinfra/modules.py
"""Qwen3 子模块手写实现（RMSNorm / RoPE / Attention / MLP / DecoderLayer）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x32).to(dtype)
```

- [x] **Step 4: 跑测试通过 + commit**

```bash
PYTHONPATH=. ~/rocm10/bin/python tests/test_modules.py   # → RMSNorm OK
git add torchinfra/modules.py tests/test_modules.py
git commit -m "P1: RMSNorm"
```

---

### Task 4: RoPE（旋转位置编码）

**Files:**
- Modify: `torchinfra/modules.py`（追加）
- Test: `tests/test_modules.py`（追加 `test_rope` 到 main）

**Interfaces:**
- Produces: `rotate_half(x: Tensor) -> Tensor`；`class RotaryEmbedding(nn.Module)`：`__init__(head_dim, max_pos, theta, device)` 建 fp32 buffer `cos/sin [max_pos, head_dim]`；`apply_qk(q, k, start: int) -> (q, k)`，q/k 形状 `[b, heads, seq, head_dim]`，返回原地语义的旋转结果（half-split 布局，与 HF Qwen3 一致）

- [x] **Step 1: 追加失败测试（手推 d=4 解析解做参考）**

```python
# 追加到 tests/test_modules.py
import math
from torchinfra.modules import RotaryEmbedding, rotate_half


def test_rope():
    # rotate_half 基本性质
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda")
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]], device="cuda"))

    # 手推解析解：head_dim=4，theta=10
    # inv_freq = [1.0, 10**(-2/4)] = [1.0, rho]，rho=10**-0.5
    # pos p 的角度：前半分量 [cos p, cos(p*rho)]（后半同角重复）
    hd, theta, max_pos = 4, 10.0, 64
    rope = RotaryEmbedding(hd, max_pos, theta, device="cuda")
    q = torch.randn(1, 2, 5, hd, device="cuda")
    k = torch.randn(1, 3, 5, hd, device="cuda")
    qr, kr = rope.apply_qk(q, k, start=7)
    rho = theta ** (-2.0 / hd)
    for p_i, p in enumerate(range(7, 12)):
        c, s = math.cos(p), math.sin(p)
        c2, s2 = math.cos(p * rho), math.sin(p * rho)
        exp = torch.stack([  # 期望：half-split 旋转公式
            q[0, 0, p_i, 0] * c - q[0, 0, p_i, 2] * s,
            q[0, 0, p_i, 1] * c2 - q[0, 0, p_i, 3] * s2,
            q[0, 0, p_i, 2] * c + q[0, 0, p_i, 0] * s,
            q[0, 0, p_i, 3] * c2 + q[0, 0, p_i, 1] * s2,
        ])
        torch.testing.assert_close(qr[0, 0, p_i].float().cpu(), exp.float().cpu(),
                                   rtol=1e-3, atol=1e-3)
    print("RoPE OK")
```

并把 `main()` 改为 `test_rms_norm(); test_rope()`。

- [x] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_modules.py`
Expected: `ImportError: cannot import name 'RotaryEmbedding'`

- [x] **Step 3: 实现**

```python
# 追加到 torchinfra/modules.py
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_pos: int, theta: float = 1e6,
                 device: str = "cuda"):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                       dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        emb = torch.cat((t[:, None] @ inv[None],) * 1, dim=-1)  # placeholder 见下
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def apply_qk(self, q: torch.Tensor, k: torch.Tensor, start: int):
        s = q.shape[-2]
        pos = torch.arange(start, start + s, device=q.device)
        cos = self.cos[pos].to(q.dtype)[None, None]
        sin = self.sin[pos].to(q.dtype)[None, None]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

（实现时删掉标注 placeholder 的那行，保留 `freqs` 两行。）

- [x] **Step 4: 跑测试通过 + commit**

```bash
PYTHONPATH=. ~/rocm10/bin/python tests/test_modules.py   # → RMSNorm OK / RoPE OK
git add torchinfra/modules.py tests/test_modules.py
git commit -m "P1: RoPE（half-split 布局 + 解析解对拍）"
```

---

### Task 5: KV cache（contiguous，单请求）

**Files:**
- Create: `torchinfra/kv_cache.py`
- Test: `tests/test_kv_cache.py`

**Interfaces:**
- Produces: `class KVCache`：`__init__(n_layers, max_seq, n_kv_heads, head_dim, dtype, device)`；`append(layer_idx, k, v, start)`（k/v 形状 `[1, n_kv, s, head_dim]`）；`read(layer_idx, upto) -> (k, v)`（`[1, n_kv, upto, head_dim]` 视图）；存储布局 `[n_layers, max_seq, n_kv, head_dim]`

- [x] **Step 1: 写失败测试**

```python
# tests/test_kv_cache.py
import torch
from torchinfra.kv_cache import KVCache


def main():
    torch.manual_seed(0)
    kv = KVCache(n_layers=2, max_seq=16, n_kv_heads=4, head_dim=8,
                 dtype=torch.bfloat16, device="cuda")
    k1 = torch.randn(1, 4, 5, 8, device="cuda", dtype=torch.bfloat16)
    v1 = torch.randn_like(k1)
    kv.append(0, k1, v1, start=0)
    k2 = torch.randn(1, 4, 3, 8, device="cuda", dtype=torch.bfloat16)
    v2 = torch.randn_like(k2)
    kv.append(0, k2, v2, start=5)
    k, v = kv.read(0, upto=8)
    assert k.shape == (1, 4, 8, 8)
    ref = torch.cat([k1, k2], dim=2).permute(0, 2, 1, 3)  # [1,4,8,8]
    torch.testing.assert_close(k, ref)
    torch.testing.assert_close(v, torch.cat([v1, v2], dim=2).permute(0, 2, 1, 3))
    # 只读前缀
    k5, _ = kv.read(0, upto=5)
    torch.testing.assert_close(k5, k1.permute(0, 2, 1, 3))
    # 第 1 层未被污染
    assert float(kv.k[1].abs().sum()) == 0.0
    print("KVCache OK")

if __name__ == "__main__":
    main()
```

- [x] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_kv_cache.py`
Expected: `ModuleNotFoundError: No module named 'torchinfra.kv_cache'`

- [x] **Step 3: 实现**

```python
# torchinfra/kv_cache.py
"""contiguous KV cache：单请求、每层预分配 [max_seq, n_kv, head_dim]。"""
import torch


class KVCache:
    def __init__(self, n_layers: int, max_seq: int, n_kv_heads: int, head_dim: int,
                 dtype=torch.bfloat16, device: str = "cuda"):
        self.k = torch.zeros(n_layers, max_seq, n_kv_heads, head_dim,
                             dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.max_seq = max_seq

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
               start: int) -> None:
        s = k.shape[2]
        self.k[layer_idx, start:start + s] = k[0].permute(1, 0, 2)
        self.v[layer_idx, start:start + s] = v[0].permute(1, 0, 2)

    def read(self, layer_idx: int, upto: int):
        k = self.k[layer_idx, :upto].unsqueeze(0).permute(0, 2, 1, 3)
        v = self.v[layer_idx, :upto].unsqueeze(0).permute(0, 2, 1, 3)
        return k, v
```

- [x] **Step 4: 跑测试通过 + commit**

```bash
PYTHONPATH=. ~/rocm10/bin/python tests/test_kv_cache.py
git add torchinfra/kv_cache.py tests/test_kv_cache.py
git commit -m "P1: contiguous KV cache"
```

---

### Task 6: Attention（GQA + QK-Norm + causal mask）

**Files:**
- Modify: `torchinfra/modules.py`（追加 `Attention`）
- Test: `tests/test_modules.py`（追加 `test_attention`）

**Interfaces:**
- Consumes: `RMSNorm`、`RotaryEmbedding.apply_qk`、`KVCache.append/read`（签名见前）
- Produces: `class Attention(nn.Module)`：`__init__(cfg: Qwen3Config)`，子模块名 `q_proj/k_proj/v_proj/o_proj/q_norm/k_norm`（都是无 bias Linear / RMSNorm）；`forward(hidden: [b,s,hidden], rope, kv: KVCache, layer_idx: int, start_pos: int) -> [b,s,hidden]`。decode（s==1）不做 causal mask。

- [x] **Step 1: 追加失败测试（两用例：无参考实现对照 + 分块等价）**

```python
# 追加到 tests/test_modules.py
from torchinfra.kv_cache import KVCache
from torchinfra.modules import Attention


def _ref_attention(hidden, attn, rope, positions):
    """独立参考：显式 eager 公式（复用 attn 的权重，不复用其 forward）。"""
    b, s, _ = hidden.shape
    n_h, n_kv, hd = attn.n_heads, attn.n_kv, attn.hd
    q = attn.q_norm(attn.q_proj(hidden).view(b, s, n_h, hd).transpose(1, 2))
    k = attn.k_norm(attn.k_proj(hidden).view(b, s, n_kv, hd).transpose(1, 2))
    v = attn.v_proj(hidden).view(b, s, n_kv, hd).transpose(1, 2)
    q, k = rope.apply_qk(q, k, start=positions[0])
    k = k.repeat_interleave(n_h // n_kv, dim=1)
    v = v.repeat_interleave(n_h // n_kv, dim=1)
    sc = (q.float() @ k.float().transpose(-1, -2)) * attn.scale
    mask = torch.arange(s, device="cuda")[None, :] > \
           torch.arange(s, device="cuda")[:, None]
    sc = sc.masked_fill(mask[None, None], float("-inf"))
    out = torch.softmax(sc, dim=-1).to(v.dtype) @ v
    return attn.o_proj(out.transpose(1, 2).reshape(b, s, n_h * hd))


def test_attention():
    from torchinfra.config import Qwen3Config
    cfg = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=4, head_dim=16, rms_norm_eps=1e-6,
                      rope_theta=10000.0, max_position_embeddings=64,
                      tie_word_embeddings=False)
    torch.manual_seed(1)
    attn = Attention(cfg).cuda().to(torch.bfloat16).eval()
    rope = RotaryEmbedding(cfg.head_dim, 64, cfg.rope_theta).cuda()

    x = torch.randn(1, 9, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    # 用例 A：一次算整段 == 独立参考
    kv = KVCache(1, 64, cfg.num_key_value_heads, cfg.head_dim)
    with torch.no_grad():
        out_full = attn(x, rope, kv, layer_idx=0, start_pos=0)
        out_ref = _ref_attention(x, attn, rope, list(range(9)))
    torch.testing.assert_close(out_full.float(), out_ref.float(),
                               rtol=1e-2, atol=1e-2)
    # 用例 B：分块（prefill 5 + 续算 4）与整段一致 —— cache 语义正确性
    kv2 = KVCache(1, 64, cfg.num_key_value_heads, cfg.head_dim)
    with torch.no_grad():
        attn(x[:, :5], rope, kv2, 0, start_pos=0)
        out_chunk = attn(x[:, 5:], rope, kv2, 0, start_pos=5)
    torch.testing.assert_close(out_chunk.float(), out_full[:, 5:].float(),
                               rtol=1e-2, atol=1e-2)
    print("Attention OK")
```

把 `test_attention()` 加入 `main()`。

- [x] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_modules.py`
Expected: `ImportError: cannot import name 'Attention'`

- [x] **Step 3: 实现**

```python
# 追加到 torchinfra/modules.py
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def forward(self, hidden, rope, kv, layer_idx, start_pos):
        b, s, _ = hidden.shape
        q = self.q_proj(hidden).view(b, s, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(hidden).view(b, s, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(hidden).view(b, s, self.n_kv, self.hd).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rope.apply_qk(q, k, start_pos)
        kv.append(layer_idx, k, v, start_pos)
        cur = start_pos + s
        kc, vc = kv.read(layer_idx, cur)
        kc = kc.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        vc = vc.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        scores = (q @ kc.transpose(-1, -2)) * self.scale
        if s > 1:  # causal mask；decode 单 token 不需要
            qpos = torch.arange(start_pos, cur, device=q.device)
            kpos = torch.arange(cur, device=q.device)
            scores = scores.masked_fill(
                kpos[None, None, None, :] > qpos[None, None, :, None],
                float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (probs @ vc).transpose(1, 2).reshape(b, s, self.n_heads * self.hd)
        return self.o_proj(out)
```

- [x] **Step 4: 跑测试通过 + commit**

```bash
PYTHONPATH=. ~/rocm10/bin/python tests/test_modules.py
git add torchinfra/modules.py tests/test_modules.py
git commit -m "P1: GQA attention（QK-Norm + causal + cache 读写）"
```

---

### Task 7: MLP + DecoderLayer + 整模型（HF 对拍）

**Files:**
- Modify: `torchinfra/modules.py`（追加 `MLP`、`DecoderLayer`）
- Create: `torchinfra/model.py`
- Create: `tests/_ref.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: Task 3-6 全部；`load_weights`（Task 2）
- Produces: `class MLP`（gate/up/down，SiLU 门控）；`class DecoderLayer`（input_layernorm → attn → 残差 → post_attention_layernorm → mlp → 残差）；`class Qwen3Model`（embed_tokens/layers/norm）；`class Qwen3ForCausalLM`：`__init__(cfg, max_seq=8192)`，`forward(input_ids, kv, start_pos, logits_to_keep: "last"|"all") -> logits [b, s 或 1, vocab]`，类方法 `from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16, max_seq=8192)`；属性 `cfg`、`max_seq`、`rope`

- [x] **Step 1: 写共享参考工具 tests/_ref.py**

```python
# tests/_ref.py
"""测试共享：HF 参考模型懒加载（进程内只加载一次）+ 固定 prompt。"""
from functools import lru_cache
from pathlib import Path
import torch

MODEL_DIR = Path.home() / "models" / "Qwen3-0.6B"
PROMPT = "简单解释一下什么是paged attention。"

@lru_cache(maxsize=2)
def hf_model(dtype: str = "bfloat16") -> "torch.nn.Module":
    from transformers import AutoModelForCausalLM
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dt)
    return m.cuda().eval()

@lru_cache(maxsize=1)
def hf_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)
```

- [x] **Step 2: 写失败测试 tests/test_model.py**

```python
# tests/test_model.py
import torch
from tests._ref import MODEL_DIR, PROMPT, hf_model, hf_tokenizer
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.kv_cache import KVCache


def main():
    tok = hf_tokenizer()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    assert ids.shape[1] >= 16

    # bf16 整模型 logits 对拍（公开 API，内部实现不依赖）
    ours = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    kv = KVCache(ours.cfg.num_hidden_layers, ours.max_seq,
                 ours.cfg.num_key_value_heads, ours.cfg.head_dim)
    with torch.no_grad():
        lo = ours(ids, kv=kv, start_pos=0, logits_to_keep="all").float()
    with torch.no_grad():
        lh = hf_model("bfloat16")(ids).logits.float()
    top1_match = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    assert top1_match >= 0.9, f"top-1 一致率过低: {top1_match}"
    assert (lo - lh).abs().max().item() < 1.0

    # 分块等价：prefill 前 16 token + decode 式逐 token，末位 logits 应一致
    kv2 = KVCache(ours.cfg.num_hidden_layers, ours.max_seq,
                  ours.cfg.num_key_value_heads, ours.cfg.head_dim)
    n = 16
    with torch.no_grad():
        ours(ids[:, :n], kv=kv2, start_pos=0, logits_to_keep="last")
        for i in range(n, ids.shape[1]):
            last = ours(ids[:, i - 1:i], kv=kv2, start_pos=i - 1,
                        logits_to_keep="last")
        chunked_last = last.float()
    torch.testing.assert_close(chunked_last, lo[:, -1], rtol=5e-2, atol=5e-2)
    print(f"model OK: top-1 match {top1_match:.3f}")

if __name__ == "__main__":
    main()
```

- [x] **Step 3: 跑测试确认失败**

Run: `PYTHONPATH=. ~/rocm10/bin/python tests/test_model.py`
Expected: `ModuleNotFoundError: No module named 'torchinfra.model'`

- [x] **Step 4: 实现 MLP / DecoderLayer（追加到 modules.py）**

```python
# 追加到 torchinfra/modules.py
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, rope, kv, layer_idx, start_pos):
        x = x + self.self_attn(self.input_layernorm(x), rope, kv, layer_idx, start_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
```

- [x] **Step 5: 实现 model.py**

```python
# torchinfra/model.py
"""Qwen3 整模型：HF 参数名 + from_pretrained。"""
from pathlib import Path
import torch
import torch.nn as nn
from torchinfra.config import Qwen3Config
from torchinfra.modules import RMSNorm, RotaryEmbedding, DecoderLayer
from torchinfra.kv_cache import KVCache
from torchinfra.weights import load_weights


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids, rope, kv, start_pos):
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, rope, kv, i, start_pos)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config, max_seq: int = 8192):
        super().__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope = None  # from_pretrained 里在 dtype 转换之后创建（保 fp32）

    def forward(self, input_ids, kv: KVCache, start_pos: int = 0,
                logits_to_keep: str = "last"):
        h = self.model(input_ids, self.rope, kv, start_pos)
        h = h[:, -1:] if logits_to_keep == "last" else h
        return self.lm_head(h)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device: str = "cuda",
                        dtype=torch.bfloat16, max_seq: int = 8192):
        cfg = Qwen3Config.from_json(model_dir)
        m = cls(cfg, max_seq=max_seq).to(device=device)
        m = m.to(dtype)
        # rope 的 cos/sin 必须保 fp32（大位置角对精度敏感），所以在 cast 之后创建
        m.rope = RotaryEmbedding(cfg.head_dim, max_seq, cfg.rope_theta,
                                 device=device)
        sd = load_weights(model_dir, device=device, dtype=dtype)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if cfg.tie_word_embeddings:
            assert set(missing) == {"lm_head.weight"} and not unexpected, \
                f"权重映射异常: missing={missing}, unexpected={unexpected}"
            m.lm_head.weight = m.model.embed_tokens.weight
        else:
            assert not missing and not unexpected
        m.eval()
        return m
```

- [x] **Step 6: 跑测试通过（这是最容易暴露 bug 的环节，备好排查路径）+ commit**

若 top-1 一致率低，按序排查：causal mask off-by-one → RoPE 布局（half vs interleaved）→ QK-Norm 位置（projection 之后、RoPE 之前）→ 缓存写入的 permute。

```bash
PYTHONPATH=. ~/rocm10/bin/python tests/test_model.py
git add torchinfra/modules.py torchinfra/model.py tests/_ref.py tests/test_model.py
git commit -m "P1: MLP/DecoderLayer/整模型 + HF logits 对拍"
```

---

### Task 8: tokenizer 封装

**Files:**
- Create: `torchinfra/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces: `class TokenizerBox`：`__init__(model_dir)`；`encode(text) -> list[int]`；`decode(ids: list[int]) -> str`；`apply_chat(messages: list[dict], add_generation_prompt=True) -> list[int]`（`enable_thinking=False`；若模板不支持该参数则去掉重试）；`eos_ids: set[int]`（取 `eos_token_id` ∪ `generation_config` 里的值）

- [x] **Step 1: 写失败测试**

```python
# tests/test_tokenizer.py
from pathlib import Path
from torchinfra.tokenizer import TokenizerBox

MODEL_DIR = Path.home() / "models" / "Qwen3-0.6B"


def main():
    tb = TokenizerBox(MODEL_DIR)
    s = "你好，world！Paged attention 是什么？"
    ids = tb.encode(s)
    assert len(ids) > 5
    assert tb.decode(ids) == s
    chat = tb.decode(tb.apply_chat([{"role": "user", "content": "你好"}]))
    assert "<|im_start|>user" in chat and "<|im_end|>" in chat
    assert chat.endswith("<|im_start|>assistant")
    assert isinstance(tb.eos_ids, set) and len(tb.eos_ids) >= 1
    print("tokenizer OK, eos_ids =", tb.eos_ids)

if __name__ == "__main__":
    main()
```

- [x] **Step 2: 确认失败 → Step 3: 实现**

```python
# torchinfra/tokenizer.py
"""HF tokenizer 薄封装：encode/decode/chat 模板/eos。"""
from pathlib import Path


class TokenizerBox:
    def __init__(self, model_dir: str | Path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        eos = {self.tok.eos_token_id}
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(model_dir)
            if gc.eos_token_id is not None:
                ids = gc.eos_token_id
                eos |= set(ids) if isinstance(ids, (list, tuple)) else {ids}
        except Exception:
            pass
        self.eos_ids = {int(e) for e in eos if e is not None}

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def apply_chat(self, messages: list[dict],
                   add_generation_prompt: bool = True) -> list[int]:
        try:
            return self.tok.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt,
                tokenize=True, enable_thinking=False)
        except TypeError:  # 模板不支持 enable_thinking
            return self.tok.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt,
                tokenize=True)
```

- [x] **Step 4: 跑测试通过 + commit**

```bash
PYTHONPATH=. ~/rocm10/bin/python tests/test_tokenizer.py
git add torchinfra/tokenizer.py tests/test_tokenizer.py
git commit -m "P1: tokenizer 封装（chat 模板关 thinking）"
```

---

### Task 9: 采样器

**Files:**
- Create: `torchinfra/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Produces: `sample_token(logits: Tensor[vocab] fp, temperature=0.0, top_k=0, top_p=0.0, generator: torch.Generator | None = None) -> int`；`temperature <= 0` 时 greedy；top_k/top_p 用 HF 风格截断（`cumsum - prob <= top_p`，保证至少留 1 个）

- [x] **Step 1: 写失败测试**

```python
# tests/test_sampling.py
import torch
from torchinfra.sampling import sample_token


def main():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5], device="cuda")
    assert sample_token(logits) == 1                      # greedy
    assert sample_token(logits, temperature=0.0) == 1
    assert sample_token(logits, temperature=1e-9) == 1    # 趋近 greedy
    assert sample_token(logits, top_k=1, temperature=1.0) == 1

    # top_p 支撑集：probs≈[.24,.47,.20,.09] 排序后 cumsum=[.47,.67,.87,1]
    # top_p=0.5 按保留规则 cumsum-prob<=p → 只留 idx1
    for seed in range(20):
        assert sample_token(logits, temperature=1.0, top_p=0.5,
                            generator=torch.Generator(device="cuda").manual_seed(seed)) == 1
    # top_p=0.99 留 idx1、idx3（.47<=.99, .67-.24=.43<=.99, .87-.20=.67<=.99 → 前3个）
    g = torch.Generator(device="cuda").manual_seed(0)
    picks = {sample_token(logits, temperature=1.0, top_p=0.99, generator=g)
             for _ in range(300)}
    assert picks <= {1, 3, 2}, picks

    # 种子可复现
    g1 = torch.Generator(device="cuda").manual_seed(42)
    g2 = torch.Generator(device="cuda").manual_seed(42)
    a = [sample_token(logits, temperature=1.0, generator=g1) for _ in range(10)]
    b = [sample_token(logits, temperature=1.0, generator=g2) for _ in range(10)]
    assert a == b
    print("sampling OK")

if __name__ == "__main__":
    main()
```

（注：具体支撑集断言以 softmax 实际值微调，先跑一遍打印 `torch.softmax` 确认再定死断言。）

- [x] **Step 2: 确认失败 → Step 3: 实现**

```python
# torchinfra/sampling.py
"""采样：greedy / temperature / top-k / top-p（HF 风格截断）。"""
import torch


def sample_token(logits: torch.Tensor, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 0.0,
                 generator: torch.Generator | None = None) -> int:
    logits = logits.float()
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        keep = (probs.cumsum(-1) - probs) <= top_p  # 首个越界 token 仍保留
        logits = torch.full_like(logits, float("-inf"))
        logits[idx[keep]] = sorted_logits[keep]
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
```

- [x] **Step 4: 跑测试通过 + commit**

```bash
PYTHONPATH=. ~/rocm10/bin/python tests/test_sampling.py
git add torchinfra/sampling.py tests/test_sampling.py
git commit -m "P1: 采样器（greedy/temp/top-k/top-p）"
```

---

### Task 10: 生成循环 + chat CLI + 终局验收

**Files:**
- Create: `torchinfra/generate.py`
- Create: `torchinfra/chat.py`
- Test: `tests/test_p1_acceptance.py`

**Interfaces:**
- Consumes: 全部前序
- Produces: `generate(model, tok: TokenizerBox, prompt_ids, max_new_tokens=128, temperature=0.0, top_k=0, top_p=0.0, seed=None) -> Iterator[tuple[int, str]]`（流式 yield `(token_id, 增量文本)`，遇 EOS 停）；`chat.py` 的 `main()`（argparse：`--model`、`--max-new`、`--temperature`、`--top-p`、`--seed`）

- [x] **Step 1: 写终局验收测试（P1 金标准：fp32 双侧 greedy 逐 token 一致）**

```python
# tests/test_p1_acceptance.py
"""P1 终局验收：与 transformers 同 prompt 同参数生成逐 token 一致。"""
import torch
from tests._ref import MODEL_DIR, PROMPT, hf_model, hf_tokenizer
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.tokenizer import TokenizerBox
from torchinfra.generate import generate


def main():
    N = 32
    tb = TokenizerBox(MODEL_DIR)
    ids = tb.encode(PROMPT)

    ours = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    ours_ids = [t for t, _ in generate(ours, tb, ids, max_new_tokens=N)]

    hf = hf_model("float32")
    with torch.no_grad():
        out = hf.generate(torch.tensor([ids], device="cuda"),
                          max_new_tokens=N, do_sample=False,
                          pad_token_id=tb.tok.eos_token_id)
    hf_ids = out[0, len(ids):].tolist()

    assert ours_ids == hf_ids, \
        f"不一致:\nours={ours_ids}\nhf  ={hf_ids}"
    print(f"acceptance OK: {N} tokens 逐 token 一致")
    print(tb.decode(ids + ours_ids)[:400])

if __name__ == "__main__":
    main()
```

- [x] **Step 2: 确认失败（No module named generate）→ Step 3: 实现 generate.py**

```python
# torchinfra/generate.py
"""prefill + decode 两阶段流式生成。"""
from typing import Iterator
import torch
from torchinfra.kv_cache import KVCache
from torchinfra.sampling import sample_token


@torch.no_grad()
def generate(model, tok, prompt_ids: list[int], max_new_tokens: int = 128,
             temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
             seed: int | None = None) -> Iterator[tuple[int, str]]:
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
    dtype = next(model.parameters()).dtype
    kv = KVCache(model.cfg.num_hidden_layers, model.max_seq,
                 model.cfg.num_key_value_heads, model.cfg.head_dim,
                 dtype=dtype)
    ids = list(prompt_ids)
    logits = model(torch.tensor([ids], device="cuda"), kv=kv, start_pos=0,
                   logits_to_keep="last")
    pos = len(ids)
    text = ""
    for _ in range(max_new_tokens):
        t = sample_token(logits[0, -1], temperature, top_k, top_p, gen)
        if t in tok.eos_ids:
            break
        ids.append(t)
        new_text = tok.decode(ids)      # 简单起见整段重解码，O(n²) 可接受
        delta, text = new_text[len(text):], new_text
        yield t, delta
        if pos + 1 >= model.max_seq:
            break
        logits = model(torch.tensor([[t]], device="cuda"), kv=kv,
                       start_pos=pos, logits_to_keep="last")
        pos += 1
```

- [x] **Step 4: 实现 chat.py**

```python
# torchinfra/chat.py
"""交互对话 CLI：python -m torchinfra.chat --model ~/models/Qwen3-0.6B"""
import argparse
import time
from pathlib import Path
import torch
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.tokenizer import TokenizerBox
from torchinfra.generate import generate

SYSTEM = "你是一个乐于助人的中文助手，回答简洁准确。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "Qwen3-0.6B"))
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = Qwen3ForCausalLM.from_pretrained(args.model)
    tok = TokenizerBox(args.model)
    messages = [{"role": "system", "content": SYSTEM}]
    print("模型就绪，输入文字开始对话（Ctrl-D 退出）")
    while True:
        try:
            user = input("\n用户> ").strip()
        except EOFError:
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        ids = tok.apply_chat(messages)
        n_in = len(ids)
        t0 = time.perf_counter()
        n_out = 0
        print("助手> ", end="", flush=True)
        for t, delta in generate(model, tok, ids, args.max_new,
                                 args.temperature, top_p=args.top_p,
                                 seed=args.seed):
            print(delta, end="", flush=True)
            n_out += 1
        dt = time.perf_counter() - t0
        print(f"\n[{n_out} tokens / {dt:.1f}s = {n_out / dt:.1f} tok/s]"
              f"（prefill {n_in}）")
        messages.append({"role": "assistant",
                         "content": tok.decode(
                             [t for t, _ in generate.__wrapped__()] ) if False else ""})
        # 注：多轮上下文简化处理——每轮重建全部历史，assistant 内容留空即可工作，
        # 完整做法是把生成 ids 拼回 messages（实现时补上，见下方说明）


if __name__ == "__main__":
    main()
```

**说明（实现时执行）**：上面多轮历史处理是刻意标注的简化，实现时改为——在 generate 外记录本轮流式生成的 ids（把 `generate` 的 yield 累积成 `turn_ids`），`messages.append({"role": "assistant", "content": tok.decode(turn_ids)})`，删掉那行占位的三元表达式。

- [x] **Step 5: 跑终局验收**

```bash
PYTHONPATH=. ~/rocm10/bin/python tests/test_p1_acceptance.py
```

Expected: `acceptance OK: 32 tokens 逐 token 一致`。若 bf16/fp32 有分歧：两侧都已 fp32，分歧只可能来自实现 bug（对照 Task 7 的排查清单）或 HF generate 的 eos/pad 细节（检查 `out[0, len(ids):]` 长度是否恰为 N）。

- [x] **Step 6: 手动验收 chat（记录 tokens/s 到 README）**

```bash
PYTHONPATH=. ~/rocm10/bin/python -m torchinfra.chat
```

问一句"用三句话介绍一下你自己"，记录输出与 tok/s；README 的 P1 勾选框打勾并附数字。

- [x] **Step 7: 全量测试回归 + commit**

```bash
for t in tests/test_config.py tests/test_weights.py tests/test_modules.py \
         tests/test_kv_cache.py tests/test_model.py tests/test_tokenizer.py \
         tests/test_sampling.py tests/test_p1_acceptance.py; do
  PYTHONPATH=. ~/rocm10/bin/python $t || exit 1
done
git add -A
git commit -m "P1: 生成循环 + chat CLI + 终局验收（32 token 逐 token 一致）"
```

---

## Self-Review 记录

- **Spec 覆盖**：safetensors 加载(T2)、tokenizer 借 HF(T8)、手写前向全件(T3-T7)、contiguous KV cache(T5)、两阶段(T10)、采样四级(T9)、CLI(T10)、三条验收(T3-T7 对拍 / T10 fp32 逐 token / T10 Step6 手动)——全覆盖
- **占位符**：Task 4 代码中标注了一行待删除的 placeholder 行（刻意保留为教学注记，Step 3 文字已说明）；Task 10 chat.py 的多轮历史占位同理有明确处理指令。除此之外无 TBD
- **类型一致性**：`KVCache.append(k,v: [1,nkv,s,d])` 在 Task 5/6/10 三处调用一致；`from_pretrained` 返回带 `.cfg/.max_seq/.rope` 属性在 T7 定义、T10 消费；`TokenizerBox.eos_ids: set[int]` 在 T8 定义、T10 使用
