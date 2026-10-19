# P2 Triton 自定义算子 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Triton 重写引擎热点算子（add/RMSNorm/softmax/RoPE + fused RMSNorm+residual），对拍 torch 实现、autotune 记录性能，并把 Triton 算子接回引擎通过 P1 终局验收。

**Architecture:** 新增 `torchinfra/triton_kernels/` 包，每个算子一个文件（jit kernel + autotune 包装 + 模块化封装）。torch 实现保留为参考实现与对拍基线；引擎通过 `backend="triton"` 参数切换（`from_pretrained` 透传）。对拍策略沿用 P1：子模块级独立参考 + 整引擎级 HF 金标准（fp32 greedy 逐 token）。

**Tech Stack:** triton 3.8.0（gfx1201 已冒烟验证）、torch 2.14+rocm10.1、`triton.testing.do_bench`。

**Spec:** `docs/superpowers/specs/2026-10-19-torchinfra-design.md` §3 P2（Triton 版）

## Global Constraints

- python 一律 `~/rocm10/bin/python`，仓库根 `PYTHONPATH=.` 运行
- 测试纯脚本 assert；对拍容差 bf16 用 rtol/atol 1e-2
- Triton kernel 内部一律 fp32 累加/计算，边界按指针 dtype 转换（与 torch 参考实现同策略）
- 引擎 torch 路径不得破坏：每任务结束跑受影响回归
- `backend="triton"` 时引擎必须重新通过 test_p1_acceptance（fp32 greedy 逐 token）

## 文件结构

```
torchinfra/triton_kernels/
  __init__.py        # 导出全部包装
  add.py             # elementwise add（T1，兼 bench 框架首例）
  rmsnorm.py         # RMSNorm kernel + TritonRMSNorm 模块 + fused residual（T2/T5）
  softmax.py         # last-dim softmax（T3）
  rope.py            # TritonRoPE（T4，继承 torch RotaryEmbedding 的 buffer）
tools/bench_ops.py   # do_bench 汇总表（T7）
tests/test_triton_add.py test_triton_rmsnorm.py test_triton_softmax.py
tests/test_triton_rope.py test_p2_integration.py
docs/notes/2026-10-19-vllm-triton-kernel-notes.md   # T8
```

---

### Task 1: Triton elementwise add + bench 基础

**Files:**
- Create: `torchinfra/triton_kernels/__init__.py`、`torchinfra/triton_kernels/add.py`
- Test: `tests/test_triton_add.py`

**Interfaces:**
- Produces: `triton_add(x: Tensor, y: Tensor) -> Tensor`（bf16/fp32，1D 展平语义）；kernel `_add_kernel` 带 `@triton.autotune`

- [ ] **Step 1: 失败测试**

```python
# tests/test_triton_add.py
import torch
from torchinfra.triton_kernels import triton_add


def main():
    for dtype in (torch.bfloat16, torch.float32):
        for n in (1 << 20, 1_000_003):
            x = torch.randn(n, device="cuda", dtype=dtype)
            y = torch.randn_like(x)
            out = triton_add(x, y)
            torch.cuda.synchronize()
            torch.testing.assert_close(out.float(), (x.float() + y.float()),
                                       rtol=1e-2, atol=1e-2)
    print("triton_add OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 确认失败 → Step 3: 实现**

```python
# torchinfra/triton_kernels/add.py
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w)
             for b in (1024, 4096, 16384) for w in (4, 8)],
    key=["n"],
)
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _add_kernel[grid](x, y, out, n)
    return out
```

`__init__.py` 先只导出 `triton_add`。

- [ ] **Step 4: 通过 + commit** `P2: Triton elementwise add`

### Task 2: Triton RMSNorm

**Files:**
- Create: `torchinfra/triton_kernels/rmsnorm.py`（本任务先做纯 RMSNorm，fused 留 T5）
- Test: `tests/test_triton_rmsnorm.py`

**Interfaces:**
- Produces: `triton_rmsnorm(x: Tensor [rows, N], w: Tensor[N], eps) -> Tensor`（输入须 2D 连续或可 reshape；模块 `TritonRMSNorm(nn.Module)` 与 `modules.RMSNorm` 同参名 `weight`、同 forward 语义，非连续输入先 `.contiguous()`）

- [ ] **Step 1: 失败测试**（对拍 P1 的 torch RMSNorm：随机权重、bf16/fp32、含 N 非 2 幂）

```python
# tests/test_triton_rmsnorm.py
import torch
from torchinfra.modules import RMSNorm
from torchinfra.triton_kernels import TritonRMSNorm


def main():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        for n in (1024, 1000):
            x = torch.randn(8, 33, n, device="cuda", dtype=dtype)
            ref_m = RMSNorm(n).cuda().to(dtype)
            tri_m = TritonRMSNorm(n).cuda().to(dtype)
            with torch.no_grad():
                tri_m.weight.copy_(ref_m.weight)
            with torch.no_grad():
                a, b = ref_m(x), tri_m(x)
            torch.testing.assert_close(a.float(), b.float(), rtol=1e-2, atol=1e-2)
            assert b.dtype == dtype
    print("TritonRMSNorm OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 确认失败 → Step 3: 实现**

```python
# torchinfra/triton_kernels/rmsnorm.py
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w)
             for b in (1024, 2048) for w in (4, 8)],
    key=["N"],
)
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride, N, eps,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    rms = tl.rsqrt(tl.sum(acc, 0) / N + eps)
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rms * w
        tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty),
                 mask=mask)


class TritonRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        shape = x.shape
        x2 = x.view(-1, shape[-1])
        out = torch.empty_like(x2)
        rows, n = x2.shape
        grid = lambda meta: (rows,)
        _rmsnorm_kernel[grid](x2, self.weight, out, out.stride(0), n,
                              self.eps)
        return out.view(shape)
```

`__init__.py` 增加导出。

- [ ] **Step 4: 通过 + commit** `P2: Triton RMSNorm（对拍 torch 实现）`

### Task 3: Triton softmax（last-dim）

**Files:**
- Create: `torchinfra/triton_kernels/softmax.py`
- Test: `tests/test_triton_softmax.py`

**Interfaces:**
- Produces: `triton_softmax(x: Tensor [rows, N]) -> Tensor`；单 pass、max 减法稳定、fp32 累加；BLOCK 由 wrapper 取 `next_pow2(N)`，只 autotune num_warps（BLOCK < N 会算错，故不能进 autotune configs）

- [ ] **Step 1: 失败测试**（对拍 `torch.softmax`，含极端值 1e30 验证数值稳定 + N 非 2 幂 1000）

```python
# tests/test_triton_softmax.py
import torch
from torchinfra.triton_kernels import triton_softmax


def main():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        for n in (128, 4096, 1000):
            x = torch.randn(64, n, device="cuda", dtype=dtype) * 3
            out = triton_softmax(x)
            ref = torch.softmax(x.float(), dim=-1)
            torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-3)
    # 数值稳定：大 logit
    x = torch.tensor([[1000.0, 0.0, -1000.0, 1e30]], device="cuda")
    out = triton_softmax(x)
    ref = torch.softmax(x.float(), -1)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-6)
    print("triton_softmax OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 确认失败 → Step 3: 实现**

```python
# torchinfra/triton_kernels/softmax.py
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (2, 4, 8, 16)],
    key=["N"],
)
@triton.jit
def _softmax_kernel(x_ptr, out_ptr, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride + offs, mask=mask,
                other=float("-inf")).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(out_ptr + row * stride + offs, y.to(out_ptr.dtype.element_ty),
             mask=mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    x2 = x.view(-1, x.shape[-1])
    out = torch.empty_like(x2)
    rows, n = x2.shape
    grid = lambda meta: (rows,)
    _softmax_kernel[grid](x2, out, out.stride(0), n, BLOCK=_next_pow2(n))
    return out.view(x.shape)
```

- [ ] **Step 4: 通过 + commit** `P2: Triton softmax（max 减法稳定）`

### Task 4: Triton RoPE

**Files:**
- Create: `torchinfra/triton_kernels/rope.py`
- Test: `tests/test_triton_rope.py`

**Interfaces:**
- Consumes: `modules.RotaryEmbedding`（继承其 fp32 cos/sin buffer）
- Produces: `class TritonRoPE(RotaryEmbedding)`：`apply_qk(q, k, start)` **原地**旋转并返回（q/k 会先 `.contiguous()`——transposed 张量内存序是 [b,s,h,d]，kernel 按行主序处理；教学版接受这次拷贝，vLLM 用显式 stride 索引避免它，笔记里记）

- [ ] **Step 1: 失败测试**（对拍 torch `RotaryEmbedding.apply_qk`，bf16 + fp32，非零 start）

```python
# tests/test_triton_rope.py
import torch
from torchinfra.modules import RotaryEmbedding
from torchinfra.triton_kernels import TritonRoPE


def main():
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        ref = RotaryEmbedding(128, 512, 1e6).cuda()
        tri = TritonRoPE(128, 512, 1e6).cuda()
        q = torch.randn(1, 16, 9, 128, device="cuda", dtype=dtype)
        k = torch.randn(1, 8, 9, 128, device="cuda", dtype=dtype)
        q2, k2 = q.clone(), k.clone()
        qr_ref, kr_ref = ref.apply_qk(q.clone(), k.clone(), 7)
        qr_tri, kr_tri = tri.apply_qk(q2, k2, 7)
        torch.testing.assert_close(qr_tri.float(), qr_ref.float(),
                                   rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(kr_tri.float(), kr_ref.float(),
                                   rtol=1e-2, atol=1e-2)
    print("TritonRoPE OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 确认失败 → Step 3: 实现**

```python
# torchinfra/triton_kernels/rope.py
import torch
import triton
import triton.language as tl
from torchinfra.modules import RotaryEmbedding


@triton.jit
def _rope_inplace_kernel(x_ptr, cos_ptr, sin_ptr, S, D: tl.constexpr,
                         HALF: tl.constexpr, stride_row):
    row = tl.program_id(0)
    pos = row % S
    offs = tl.arange(0, HALF)
    base = row * stride_row
    x1 = tl.load(x_ptr + base + offs).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF + offs).to(tl.float32)
    cos = tl.load(cos_ptr + pos * D + offs)   # 表是 [max_pos, D] 双份半区，取前半
    sin = tl.load(sin_ptr + pos * D + offs)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    tl.store(x_ptr + base + offs, o1.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + base + HALF + offs, o2.to(x_ptr.dtype.element_ty))


class TritonRoPE(RotaryEmbedding):
    def apply_qk(self, q: torch.Tensor, k: torch.Tensor, start: int):
        q = q.contiguous()   # transposed 视图内存序不满足行主序假设
        k = k.contiguous()
        d = q.shape[-1]
        for x, s_len in ((q, q.shape[2]), (k, k.shape[2])):
            rows = x.numel() // d
            _rope_inplace_kernel[(rows,)](
                x, self.cos, self.sin, s_len, d, d // 2, x.stride(2),
                HALF=d // 2)
        return q, k
```

（stride_row 实为 x.stride(2)=d——行内连续；`x.stride(2)` 传 d 保持显式。）

- [ ] **Step 4: 通过 + commit** `P2: Triton RoPE（half-split 原地）`

### Task 5: fused RMSNorm + residual

**Files:**
- Modify: `torchinfra/triton_kernels/rmsnorm.py`（追加）
- Test: `tests/test_triton_rmsnorm.py`（追加 `test_fused`）

**Interfaces:**
- Produces: `fused_rmsnorm_residual(x: Tensor[rows,N], dx: Tensor, w, eps) -> (residual, normed)`：`residual = (x+dx)`（按输出 dtype 舍入后存储）、`normed = rmsnorm(residual)*w`——与"先 add 存 bf16 再 norm"的 torch 序列逐位同策略；模块 `FusedRMSNormResidual(nn.Module)`

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_triton_rmsnorm.py
from torchinfra.triton_kernels import fused_rmsnorm_residual


def test_fused():
    torch.manual_seed(1)
    for dtype in (torch.bfloat16, torch.float32):
        from torchinfra.modules import RMSNorm
        n = 1024
        x = torch.randn(4, 65, n, device="cuda", dtype=dtype)
        dx = torch.randn_like(x)
        w = torch.randn(n, device="cuda", dtype=dtype)
        # torch 参考：add（存 dtype）→ RMSNorm（fp32 内算）
        res_ref = x + dx
        norm_ref = RMSNorm(n).cuda().to(dtype)
        with torch.no_grad():
            norm_ref.weight.copy_(w)
            nref = norm_ref(res_ref)
        res, normed = fused_rmsnorm_residual(x.contiguous(), dx.contiguous(),
                                             w, 1e-6)
        torch.testing.assert_close(res.float(), res_ref.float(),
                                   rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(normed.float(), nref.float(),
                                   rtol=1e-2, atol=1e-2)
    print("fused_rmsnorm_residual OK")
```

- [ ] **Step 2: 确认失败 → Step 3: 实现（单 pass，BLOCK≥N 时一趟完成）**

```python
# 追加到 torchinfra/triton_kernels/rmsnorm.py
@triton.jit
def _fused_kernel(x_ptr, dx_ptr, w_ptr, res_ptr, norm_ptr, stride, N, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dx = tl.load(dx_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = x + dx
    r_round = r.to(res_ptr.dtype.element_ty).to(tl.float32)  # 对齐 torch 先存后读
    rms = tl.rsqrt(tl.sum(r_round * r_round, 0) / N + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(res_ptr + base + offs, r.to(res_ptr.dtype.element_ty), mask=mask)
    tl.store(norm_ptr + base + offs, (r_round * rms * w).to(norm_ptr.dtype.element_ty),
             mask=mask)


def fused_rmsnorm_residual(x: torch.Tensor, dx: torch.Tensor,
                           w: torch.Tensor, eps: float = 1e-6):
    x, dx = x.contiguous(), dx.contiguous()
    shape = x.shape
    n = shape[-1]
    res = torch.empty_like(x)
    normed = torch.empty_like(x)
    x2, dx2 = x.view(-1, n), dx.view(-1, n)
    r2, n2 = res.view(-1, n), normed.view(-1, n)
    rows = x2.shape[0]
    block = max(triton.next_power_of_2(n), 16)
    _fused_kernel[(rows,)](x2, dx2, w, r2, n2, x2.stride(0), n, eps,
                           BLOCK=block)
    return res, normed
```

注意：`res` 存的是 fp32 的 r（未舍入）还是舍入值？torch 参考的 residual 是 bf16 的——所以 `res` 也应存舍入值，`normed` 用舍入值算。上面 `res` 存了 `r`（fp32）需改为存 `r.to(dtype)`；实现时以"res 与 res_ref 逐位一致"为准修正。（TDD 会抓住这个。）

- [ ] **Step 4: 通过 + commit** `P2: fused RMSNorm+residual`

### Task 6: 引擎集成（backend="triton"）+ 终局验收

**Files:**
- Modify: `torchinfra/modules.py`（DecoderLayer 支持 fused 路径与 norm 后端选择）
- Modify: `torchinfra/model.py`（`from_pretrained(..., backend="torch"|"triton")`，triton 时构建 `TritonRoPE`）
- Test: `tests/test_p2_integration.py`

**Interfaces:**
- Produces: `DecoderLayer.__init__(cfg, norm_backend="torch")`；`Qwen3ForCausalLM.from_pretrained(model_dir, ..., backend="torch")`；triton 路径：norm 全走 `TritonRMSNorm`，残差加法走 `fused_rmsnorm_residual`，rope 用 `TritonRoPE`；attention 内 softmax 保持 torch（融合进 attention kernel 是 P3 范围）

- [ ] **Step 1: 失败测试**

```python
# tests/test_p2_integration.py
import torch
from tests._ref import MODEL_DIR, PROMPT, hf_model, hf_tokenizer
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.kv_cache import KVCache
from torchinfra.generate import generate


def main():
    tok = hf_tokenizer()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    ours = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                            backend="triton")
    kv = KVCache(ours.cfg.num_hidden_layers, ours.max_seq,
                 ours.cfg.num_key_value_heads, ours.cfg.head_dim)
    with torch.no_grad():
        lo = ours(ids, kv=kv, start_pos=0, logits_to_keep="all").float()
    with torch.no_grad():
        lh = hf_model("bfloat16")(ids).logits.float()
    top1 = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    assert top1 >= 0.9, top1
    print(f"triton 引擎 logits 对拍 top-1 = {top1:.3f}")

    # 终局：fp32 + triton 后端，greedy 与 HF 逐 token 一致
    ours32 = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                              backend="triton")
    from torchinfra.tokenizer import TokenizerBox
    tb = TokenizerBox(MODEL_DIR)
    pids = tb.encode(PROMPT)
    ours_ids = [t for t, _ in generate(ours32, tb, pids, max_new_tokens=16)]
    with torch.no_grad():
        out = hf_model("float32").generate(
            torch.tensor([pids], device="cuda"), max_new_tokens=16,
            do_sample=False, pad_token_id=tb.tok.eos_token_id)
    assert ours_ids == out[0, len(pids):].tolist()
    print("triton 引擎 fp32 greedy 16 token 与 HF 逐 token 一致")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 确认失败 → Step 3: 实现**

modules.py 的 `DecoderLayer.__init__` 加 `norm_backend` 参数；triton 时 `input_layernorm/post_attention_layernorm` 用 `TritonRMSNorm`（import 放函数内避免循环依赖/无 triton 环境下 torch 路径仍可用）。forward 分支：

```python
def forward(self, x, rope, kv, layer_idx, start_pos):
    if self._fused:  # triton
        from torchinfra.triton_kernels import fused_rmsnorm_residual
        normed = self.input_layernorm(x)          # 纯 norm 喂 attn
        attn_out = self.self_attn(normed, rope, kv, layer_idx, start_pos)
        x, normed2 = fused_rmsnorm_residual(x, attn_out,
                                            self.post_attention_layernorm.weight,
                                            self.post_attention_layernorm.eps)
        x = x + self.mlp(normed2)
    else:
        x = x + self.self_attn(self.input_layernorm(x), rope, kv, layer_idx,
                               start_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
    return x
```

model.py：`Qwen3Model.__init__(cfg, norm_backend)`、`Qwen3ForCausalLM.__init__(cfg, max_seq, norm_backend="torch")`；`from_pretrained(..., backend="torch")` → `norm_backend=backend`，且 triton 时 `m.rope = TritonRoPE(...)`（import 局部）。model 末尾 `self.norm` 同样按后端选择。

- [ ] **Step 4: 通过（含 P1 回归：test_model.py、test_p1_acceptance.py 用默认 torch 后端）+ commit** `P2: 引擎接入 Triton 后端（norm/rope/fused residual）`

### Task 7: benchmark 汇总

**Files:**
- Create: `tools/bench_ops.py`

**Interfaces:**
- Produces: 一张 markdown 表：每算子 triton vs torch（及 fused vs 未融合）在代表性 shape 下的 `do_bench` 毫秒与加速比；结果粘贴进 README

- [ ] **Step 1: 实现**（shape 取引擎真实值：rmsnorm/fused 用 [1,4096,1024] bf16；softmax [32,4096]；rope q[1,16,4096,128]+k[1,8,4096,128]；add 16M 元素）

```python
# tools/bench_ops.py
"""P2 算子基准：triton.testing.do_bench，输出 markdown 表。"""
import torch
import triton
from torchinfra.modules import RMSNorm, RotaryEmbedding
from torchinfra.triton_kernels import (TritonRMSNorm, TritonRoPE,
                                       triton_add, triton_softmax,
                                       fused_rmsnorm_residual)

def bench(fn):
    return triton.testing.do_bench(fn)  # ms

def main():
    rows = []
    torch.manual_seed(0)
    # RMSNorm
    x = torch.randn(1, 4096, 1024, device="cuda", dtype=torch.bfloat16)
    m1, m2 = RMSNorm(1024).cuda().to(torch.bfloat16), TritonRMSNorm(1024).cuda().to(torch.bfloat16)
    rows.append(("RMSNorm [1,4096,1024] bf16", bench(lambda: m1(x)),
                 bench(lambda: m2(x))))
    # softmax
    s = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16) * 3
    rows.append(("softmax [32,4096]", bench(lambda: torch.softmax(s.float(), -1)),
                 bench(lambda: triton_softmax(s))))
    # RoPE（apply_qk 含 contiguous 拷贝，公平对比同样算拷贝的 torch 版）
    r1 = RotaryEmbedding(128, 8192, 1e6).cuda()
    r2 = TritonRoPE(128, 8192, 1e6).cuda()
    q = torch.randn(1, 16, 512, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
    qc, kc = q.clone(), k.clone()
    rows.append(("RoPE q16/k8×512×128", bench(lambda: r1.apply_qk(q, k, 0)),
                 bench(lambda: r2.apply_qk(qc.contiguous(), kc.contiguous(), 0))))
    # add
    a = torch.randn(1 << 24, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    rows.append(("add 16M bf16", bench(lambda: a + b), bench(lambda: triton_add(a, b))))
    # fused vs 未融合（add + rmsnorm 两步）
    w = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    dx = torch.randn_like(x)
    def unfused():
        r = x + dx
        return RMSNorm(1024).cuda().to(torch.bfloat16)(r)
    def fused():
        fused_rmsnorm_residual(x, dx, w)
    rows.append(("residual+RMSNorm fused", bench(unfused), bench(fused)))

    print("| 算子 | torch ms | triton ms | 加速比 |")
    print("|---|---|---|---|")
    for name, t_ms, tri_ms in rows:
        print(f"| {name} | {t_ms:.3f} | {tri_ms:.3f} | {t_ms / tri_ms:.2f}x |")

if __name__ == "__main__":
    main()
```

（unfused 里每次构建 RMSNorm 不公平——实现时把模块提出闭包外。）

- [ ] **Step 2: 跑通并把表粘进 README + commit** `P2: 算子基准表`

### Task 8: vLLM/SGLang Triton kernel 阅读笔记

**Files:**
- Create: `docs/notes/2026-10-19-vllm-triton-kernel-notes.md`

- [ ] **Step 1:** 用 GitHub 工具读 vLLM 的 `vllm/attention/triton_backend/flash_attention_kernels.py`（或 SGLang 等价 Triton attention kernel）
- [ ] **Step 2:** 写笔记（≥400 字）：grid/block 划分方式、paged KV 的 block table 索引方法、causal mask 的处理、与我们 P2/P3 kernel 的对应关系、至少 3 个可借鉴的技巧（如 `tl.multiple_of`/`max_contiguous`、exp2 代替 exp、软件流水）
- [ ] **Step 3: commit** `P2: vLLM Triton attention kernel 阅读笔记`

---

## Self-Review 记录

- **Spec 覆盖**：≥3 算子对拍+autotune（T1/T2/T3/T4 四个）✓；≥1 fused 可测收益（T5+T7 表）✓；阅读笔记（T8）✓；引擎替换（T6 + P1 验收重跑）✓
- **占位符**：T5 与 T7 各有一处"实现时修正"的明确指令（residual 舍入语义、闭包外提模块），其余无
- **类型一致性**：`TritonRMSNorm(dim, eps)` 与 `RMSNorm` 构造签名一致；`TritonRoPE.apply_qk(q,k,start)` 与 torch 版签名一致；`fused_rmsnorm_residual(x,dx,w,eps)->(res,normed)` 在 T5 定义 T6 消费
