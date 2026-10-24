"""Nemotron-3.5-Rerank-8B：ministral3 双向塔 + mean-pool + 标量 score 头。

对照 transformers 5.12.1 ministral3 + NVIDIA model.py 逐条移植：
- 34 层，hidden 4096，32Q/8KV × head_dim 128，MLP 14336 SwiGLU，RMSNorm eps 1e-5
- 全双向注意力（config.is_causal=False → create_causal_mask 退化为 padding mask，
  无 padding 时 SDPA 不带 mask 也不带 is_causal）
- yarn rope：factor 16、original_max_position_embeddings 16384、base 1e6、
  beta 32/1；mscale=mscale_all_dim=1 → attention_factor=1，cos/sin 无缩放
- llama-4 式 Q attn-scale：q *= 1+0.1·ln(1+floor(pos/16384))，pos<16384 恒为 1
  （32K 段 1+0.1·ln2≈1.0693，只在上 16K 生效，kernel 里用 constexpr 关掉）
- 打分：final RMSNorm → valid mean-pool → fp32 Linear(4096→1)（score 头 bf16 权重）
- 权重走 torchinfra fp8 checkpoint，**per-projection scale**（q/k/v、gate/up
  各自独立，split_scales 产物）但仍是融合单 GEMM：epilogue 的 tensorwise
  scale_b 取 q/gate 段，k/v/up 段的比值修正（sk/sq 等）在下游自有 kernel
  （rope/silu）load 后 fp32 补乘——数学上等价 GEMM epilogue 按列 descale，
  却不掉进慢 1.66x 的 rowwise _scaled_mm 路径。共享 scale checkpoint 的
  修正恒为 1，兼容加载。视觉塔与 projector 不加载（省 0.8GB）

性能设计（沿用 P10–P12 结论）：
- 注意力核心走 SDPA：双向无 mask 时 flash 是这张卡的天花板（32K 实测 146 TF）；
  自研 Triton 只做融合类 kernel（rope+transpose、rmsnorm/silu+amax、转置+amax）
- 全部 GEMM = fp8 _scaled_mm，激活量化由生产者 kernel 吐 partial（bit-exact 融合）
- MLP 沿 S 分块（S>8192 时），压 gu [S,28672] 的 1.9GB 峰值
- B=1 小 S 走 CUDA Graph 分桶（右 padding + key mask，mask.sum 参与池化，
  replay 时随 mask 内容变化，不烘焙进图）
"""
from pathlib import Path
import json
import math
import os
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
# 32K 长序列的临时张量块大（qkv 384MB / gu 448MB），默认分配器按固定尺寸
# 切块会产生数百 MB 碎片导致 OOM；expandable_segments 让块可增长复用
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# TunableOp 现场调优（默认开）会在首次 forward 时对每个新 GEMM 形状跑
# 数十轮 hipBLASLt 基准：34 层 × 4 形状在 CPU 上单核转几分钟，且调优
# kernel 混进图捕获直接 HSA_STATUS_ERROR_MEMORY_FAULT（P11 在 jina 上
# 踩过同款）。默认只用已有结果文件/启发式；要做调优跑时显式设为 1。
os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "0")
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torchinfra.fp8_ckpt import is_fp8_checkpoint
from torchinfra.triton_kernels import triton_rmsnorm, triton_rmsnorm_amax
from torchinfra.triton_kernels.silu import silu_mul_amax
from torchinfra.fp8_linear import fp8_linear_q
from torchinfra.jina_rerank import transpose_amax

LAYER_PREFIX = "model.language_model.layers"
N_LAYERS = 34
HIDDEN = 4096
NQ, NKV, HD = 32, 8, 128
FFN = 14336
EPS = 1e-5
ROPE_BASE = 1e6
ROPE_FACTOR = 16.0
ROPE_ORIG = 16384
MAX_POS = 32768
# rope kernel 行块（同 jina QKR：TR=8 在 gfx1201 有算错前科，锁 16）
ROPE_TR = 16
ROPE_WARPS = 8
# MLP 分块：gu [S,28672] bf16 在 S=32768 时 1.9GB，超阈值按块算
MLP_CHUNK_S = 8192


def yarn_inv_freq(device="cuda") -> torch.Tensor:
    """transformers _compute_yarn_parameters 的逐行移植（[64] fp32）。

    dim<20 外推（不缩放）、dim>37 内插（÷16）、中间线性 ramp；
    truncate 默认 True → low/high 取 floor/ceil。"""
    dim = HD
    # 与 transformers 逐位一致：arange int64 → float32，全程 float32（不用 float64）
    pos_freqs = ROPE_BASE ** (torch.arange(0, dim, 2).to(torch.float32) / dim)
    inv_ext = 1.0 / pos_freqs
    inv_int = 1.0 / (ROPE_FACTOR * pos_freqs)

    def corr(r):
        return dim * math.log(ROPE_ORIG / (r * 2 * math.pi)) / (2 * math.log(ROPE_BASE))

    low = max(math.floor(corr(32)), 0)
    high = min(math.ceil(corr(1)), dim // 2 - 1)
    ramp = torch.clamp((torch.arange(dim // 2).to(torch.float32) - low)
                       / (high - low), 0, 1)
    inv = inv_int * ramp + inv_ext * (1 - ramp)
    return inv.float().to(device)


def rope_tables(max_pos: int = MAX_POS, device="cuda"):
    inv = yarn_inv_freq(device)
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)          # [max_pos, 64]
    return freqs.cos().contiguous(), freqs.sin().contiguous()


@triton.jit
def _rope_qkv_kernel(
    QKV, COS, SIN, POS, QO, KO, VO,
    s_row, s_cos, n_rows, seq, kc, vc,
    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr,
    HALF: tl.constexpr, TR: tl.constexpr, L4SC: tl.constexpr,
    ORIG: tl.constexpr, BETA: tl.constexpr,
):
    """yarn rotate-half RoPE + [B*S, (NQ+2NKV)*D] → [B,H,S,D] 转置一趟完成。

    无 qk-norm（ministral3 没有）。L4SC=True 时 Q 乘 llama-4 attn-scale
    （1+BETA·ln(1+floor(pos/ORIG))，pos<ORIG 恒 1）。V 顺带搬运省一次 permute。

    kc/vc 是 per-projection scale 的段修正：融合 GEMM 的 epilogue 只能带
    一个 tensorwise scale_b（取 q 段的 sq），k/v 段在量化时用了自己的
    sk/sv，输出差 (sk/sq)、(sv/sq) 倍——在这里 load 后 fp32 补乘，等价于
    GEMM epilogue 按列 descale，但不用掉进慢 1.66x 的 rowwise 路径。
    共享 scale checkpoint 的 kc=vc=1。
    """
    rb = tl.program_id(0)
    h = tl.program_id(1)
    rows = rb * TR + tl.arange(0, TR)
    rm = rows < n_rows
    bi = rows // seq
    row = rows - bi * seq
    off = tl.arange(0, HALF)
    pos = tl.load(POS + rows, mask=rm, other=0)
    cs = pos[:, None] * s_cos + off[None, :]
    cos = tl.load(COS + cs, mask=rm[:, None], other=0.0)
    sin = tl.load(SIN + cs, mask=rm[:, None], other=0.0)

    qb = rows[:, None] * s_row + h * D
    a = tl.load(QKV + qb + off[None, :], mask=rm[:, None], other=0.0)
    b = tl.load(QKV + qb + HALF + off[None, :], mask=rm[:, None], other=0.0)
    qa = a * cos - b * sin
    qb2 = b * cos + a * sin
    if L4SC:
        qs = 1.0 + BETA * tl.log(1.0 + tl.floor(pos.to(tl.float32) / ORIG))
        qa = qa * qs[:, None]
        qb2 = qb2 * qs[:, None]
    db = ((bi[:, None] * NQ + h) * seq + row[:, None]) * D
    tl.store(QO + db + off[None, :], qa, mask=rm[:, None])
    tl.store(QO + db + HALF + off[None, :], qb2, mask=rm[:, None])

    if h < NKV:
        koff = rows[:, None] * s_row + NQ * D + h * D
        ka = tl.load(QKV + koff + off[None, :], mask=rm[:, None],
                     other=0.0).to(tl.float32) * kc
        kb = tl.load(QKV + koff + HALF + off[None, :], mask=rm[:, None],
                     other=0.0).to(tl.float32) * kc
        kdb = ((bi[:, None] * NKV + h) * seq + row[:, None]) * D
        tl.store(KO + kdb + off[None, :], ka * cos - kb * sin, mask=rm[:, None])
        tl.store(KO + kdb + HALF + off[None, :], kb * cos + ka * sin, mask=rm[:, None])
        vb = rows[:, None] * s_row + (NQ + NKV) * D + h * D
        for dd in range(0, D, HALF):
            vv = tl.load(QKV + vb + dd + off[None, :], mask=rm[:, None],
                         other=0.0).to(tl.float32) * vc
            tl.store(VO + kdb + dd + off[None, :],
                     vv.to(VO.dtype.element_ty), mask=rm[:, None])


class _Layer:
    def __init__(self):
        self.raw: dict[str, torch.Tensor] = {}


@triton.jit
def _bidir_attn_kernel(
    Q, K, V, O, PART, VLEN, sm, S,
    NQ_PER_KV: tl.constexpr, D: tl.constexpr,
    TQ: tl.constexpr, TK: tl.constexpr,
):
    """全双向注意力（右 padding），FA2 式在线 softmax + 直写 [B, S, H*D]。

    ROCm 上 SDPA 带 attn_mask 会掉 math 后端物化 [B,H,S,S]，所以凡需要
    掩码（图分桶右 padding / 多长度 batch）都走这里；无掩码路径仍用 SDPA
    flash（大 S 天花板）。VLEN [B] 是各 batch 的有效长度：列掩码
    offs_k < vlen（双向无因果项），行掩码同理；掩码行输出写 0，tile amax
    掩 0 不污染 fp8 量化。PART [B, H, cdiv(S,TQ)] fp32。
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    H = tl.num_programs(1)
    kv_h = h // NQ_PER_KV
    vlen = tl.load(VLEN + b)
    q0 = t * TQ
    offs_q = q0 + tl.arange(0, TQ)
    offs_d = tl.arange(0, D)
    qm = offs_q < vlen
    q = tl.load(Q + ((b * H + h) * S + offs_q[:, None]) * D + offs_d[None, :],
                mask=qm[:, None], other=0.0)
    m_i = tl.full([TQ], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([TQ], dtype=tl.float32)
    acc = tl.zeros([TQ, D], dtype=tl.float32)
    kv_base = (b * (H // NQ_PER_KV) + kv_h) * S
    for ks in range(0, S, TK):
        offs_k = ks + tl.arange(0, TK)
        km = offs_k < vlen
        k = tl.load(K + (kv_base + offs_k[:, None]) * D + offs_d[None, :],
                    mask=km[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm
        mask = km[None, :]
        qk = tl.where(mask, qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.where(mask, tl.exp(qk - m_new[:, None]), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + (kv_base + offs_k[:, None]) * D + offs_d[None, :],
                    mask=km[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new
    ob = (acc / l_i[:, None]).to(O.dtype.element_ty)
    # pad 行（≥vlen 或越界 S）输出记 0：pad 行必须保持全零，否则垃圾值
    # 会顺着 o_proj GEMM 流回残差流，污染 tensorwise fp8 量化的 amax
    obz = tl.where(qm[:, None], ob.to(tl.float32), 0.0)
    tl.store(O + (b * S + offs_q[:, None]) * (H * D) + h * D + offs_d[None, :],
             obz.to(O.dtype.element_ty), mask=(offs_q < S)[:, None])
    nt = tl.cdiv(S, TQ)
    tl.store(PART + (b * H + h) * nt + t, tl.max(tl.abs(obz)))


# 自研双向 kernel 的 launch 配置（沿用滑窗 kernel 同款扫描结论：64/64/w4/st1）
BIDIR_TQ = 64
BIDIR_TK = 64
BIDIR_WARPS = 4
BIDIR_STAGES = 1


def _load_tower_fp8(model_dir: Path):
    """fp8 checkpoint → (layers[34], embed, final_norm, score_w)。视觉塔跳过。"""
    from safetensors.torch import load_file
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    layers = [_Layer() for _ in range(N_LAYERS)]
    embed = final_norm = score_w = None
    per_shard: dict[str, list[str]] = {}
    for k, f in wm.items():
        per_shard.setdefault(f, []).append(k)
    for f, keys in per_shard.items():
        sd = load_file(str(model_dir / f))
        for k in keys:
            v = sd[k]
            if k.startswith(LAYER_PREFIX):
                i, rest = k.split(".layers.")[1].split(".", 1)
                layers[int(i)].raw[rest] = v
            elif k == "model.language_model.embed_tokens.weight":
                embed = v
            elif k == "model.language_model.norm.weight":
                final_norm = v
            elif k == "language_model.score.weight":
                score_w = v
            # vision_tower / multi_modal_projector → 丢弃（文本 rerank 用不到）
        del sd
    assert embed is not None and final_norm is not None and score_w is not None
    dev = "cuda"
    for i, ly in enumerate(layers):
        r = ly.raw
        # 融合 GEMM + per-projection scale：q/k/v 拼回 [6144,4096] 单 GEMM，
        # scale_b 取 q 段；k/v 段的比值修正交给 rope kernel 补乘（同理
        # gate/up）。共享 scale checkpoint 的修正恒为 1.0，同样可加载。
        qs, ks, vs = (r["self_attn.q_proj.weight_scale"],
                      r["self_attn.k_proj.weight_scale"],
                      r["self_attn.v_proj.weight_scale"])
        ly.w_qkv_fp8 = torch.cat(
            [r["self_attn.q_proj.weight"].to(dev),
             r["self_attn.k_proj.weight"].to(dev),
             r["self_attn.v_proj.weight"].to(dev)], 0).contiguous()
        ly.w_qkv_s = qs.to(dev)
        ly.kc = float(ks.item() / qs.item())
        ly.vc = float(vs.item() / qs.item())
        ly.o_w_fp8 = r["self_attn.o_proj.weight"].to(dev)
        ly.o_w_s = r["self_attn.o_proj.weight_scale"].to(dev)
        gs, us = r["mlp.gate_proj.weight_scale"], r["mlp.up_proj.weight_scale"]
        ly.w_gu_fp8 = torch.cat(
            [r["mlp.gate_proj.weight"].to(dev),
             r["mlp.up_proj.weight"].to(dev)], 0).contiguous()
        ly.w_gu_s = gs.to(dev)
        ly.uc = float(us.item() / gs.item())
        ly.down_w_fp8 = r["mlp.down_proj.weight"].to(dev)
        ly.down_w_s = r["mlp.down_proj.weight_scale"].to(dev)
        ly.ln1 = r["input_layernorm.weight"].to(dev)
        ly.ln2 = r["post_attention_layernorm.weight"].to(dev)
        ly.raw = None   # 释放 CPU 源
    return layers, embed.to(dev), final_norm.to(dev), score_w.to(dev)


class NemotronReranker:
    GRAPH_MAX = 8192

    def __init__(self, layers, embed, final_norm, score_w, tok,
                 cos, sin, max_pos: int = MAX_POS):
        self.layers = layers
        self.embed = embed
        self.final_norm = final_norm
        self.score_w = score_w          # [1, 4096] bf16
        self.tok = tok
        self.cos, self.sin = cos, sin
        self.max_pos = max_pos
        self._graphs: dict = {}
        self._graph_pool = None
        self._pos_cache: dict = {}

    @classmethod
    def from_pretrained(cls, model_dir: str | Path):
        model_dir = Path(model_dir)
        if not is_fp8_checkpoint(model_dir):
            raise ValueError("需要 torchinfra fp8 checkpoint（save_fp8_checkpoint_sharded 产物）")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        layers, embed, final_norm, score_w = _load_tower_fp8(model_dir)
        cos, sin = rope_tables()
        if hasattr(torch.cuda, "tunable"):
            torch.cuda.tunable.enable(True)
        return cls(layers, embed, final_norm, score_w, tok, cos, sin)

    # ---- 前向 ----

    def _attn_block(self, ly: _Layer, x, B, S, vlen):
        """vlen None → 无 padding 走 SDPA flash；否则右 padding 走自研双向 kernel。"""
        h, ph = triton_rmsnorm_amax(x, ly.ln1, EPS)
        qkv = fp8_linear_q(h, ly.w_qkv_fp8, ly.w_qkv_s, ph)
        cos, sin = self.cos, self.sin
        half = HD // 2
        q = torch.empty(B, NQ, S, HD, dtype=qkv.dtype, device=x.device)
        k = torch.empty(B, NKV, S, HD, dtype=qkv.dtype, device=x.device)
        v = torch.empty(B, NKV, S, HD, dtype=qkv.dtype, device=x.device)
        # kernel 按 [B*S, C] 展平行寻址：必须传展平后的行跨度 C=6144
        # （qkv [B,S,C] 的 stride(0)=S*C，曾因传错越界 HSA fault）
        qkv2 = qkv.view(B * S, -1)
        _rope_qkv_kernel[(triton.cdiv(B * S, ROPE_TR), NQ)](
            qkv2, cos, sin, self.pos_cache(S, B, x.device), q, k, v,
            qkv2.stride(0), cos.stride(0), B * S, S, ly.kc, ly.vc,
            NQ=NQ, NKV=NKV, D=HD, HALF=half, TR=ROPE_TR,
            L4SC=S > ROPE_ORIG, ORIG=float(ROPE_ORIG), BETA=0.1,
            num_warps=ROPE_WARPS)
        del qkv   # 32K 时 384MB：rope 已搬完，别占着过注意力
        if vlen is None:
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            return transpose_amax(out)
        out = torch.empty(B, S, NQ * HD, dtype=q.dtype, device=q.device)
        nt = triton.cdiv(S, BIDIR_TQ)
        apart = torch.empty(B, NQ, nt, dtype=torch.float32, device=q.device)
        _bidir_attn_kernel[(B, NQ, nt)](
            q, k, v, out, apart, vlen, HD ** -0.5, S,
            NQ_PER_KV=NQ // NKV, D=HD,
            TQ=BIDIR_TQ, TK=BIDIR_TK,
            num_warps=BIDIR_WARPS, num_stages=BIDIR_STAGES)
        return out, apart

    def _mlp_block(self, ly: _Layer, x, B, S):
        """融合 gu GEMM + silu_mul（up 段修正）+ down GEMM；S 超阈值分块。"""
        if B * S <= MLP_CHUNK_S:
            h2, ph2 = triton_rmsnorm_amax(x, ly.ln2, EPS)
            gu = fp8_linear_q(h2, ly.w_gu_fp8, ly.w_gu_s, ph2)
            act, pa = silu_mul_amax(gu.view(B * S, FFN * 2), 1.0, ly.uc)
            return x + fp8_linear_q(act, ly.down_w_fp8, ly.down_w_s,
                                    pa).view(B, S, HIDDEN)
        rows = x.reshape(B * S, HIDDEN)
        out = torch.empty_like(rows)
        for r0 in range(0, B * S, MLP_CHUNK_S):
            rc = rows[r0:r0 + MLP_CHUNK_S]
            h2, ph2 = triton_rmsnorm_amax(rc, ly.ln2, EPS)
            gu = fp8_linear_q(h2, ly.w_gu_fp8, ly.w_gu_s, ph2)
            act, pa = silu_mul_amax(gu, 1.0, ly.uc)
            out[r0:r0 + MLP_CHUNK_S] = fp8_linear_q(act, ly.down_w_fp8,
                                                    ly.down_w_s, pa)
        return x + out.view(B, S, HIDDEN)

    def pos_cache(self, S: int, B: int, device) -> torch.Tensor:
        """0..S-1 重复 B 份 [B*S] int32（右 padding 下 pad 行位置无关紧要，被掩）。"""
        key = (S, B, str(device))
        p = self._pos_cache.get(key)
        if p is None:
            p = torch.arange(S, device=device, dtype=torch.int32) \
                .repeat(B).reshape(-1)
            self._pos_cache[key] = p
        return p

    @torch.no_grad()
    def forward_hidden(self, input_ids: torch.Tensor,
                       vlen: torch.Tensor | None = None):
        """input_ids [B,S]（右 padding）→ final-norm hidden [B,S,4096]。

        vlen None 表示整批无 padding（SDPA flash 路径）；否则 vlen [B] int32
        是各 batch 有效长度（自研双向 kernel 路径）。
        """
        B, S = input_ids.shape
        if S > self.max_pos:
            raise ValueError(f"S={S} 超过 rope 表容量 {self.max_pos}")
        x = self.embed[input_ids]
        if vlen is not None:
            x = x * (torch.arange(S, device=x.device)[None, :]
                     < vlen[:, None]).unsqueeze(-1).to(x.dtype)
        x = x.view(B, S, HIDDEN)
        for ly in self.layers:
            attn_out, ap = self._attn_block(ly, x, B, S, vlen)
            x = x + fp8_linear_q(attn_out, ly.o_w_fp8, ly.o_w_s, ap)
            x = self._mlp_block(ly, x, B, S)
        return triton_rmsnorm(x.view(B * S, HIDDEN), self.final_norm, EPS) \
            .view(B, S, HIDDEN)

    def _pool_score(self, h: torch.Tensor, vlen: torch.Tensor) -> torch.Tensor:
        """final hidden [B,S,4096] → logits [B] fp32（valid mean-pool + score 头）。"""
        m = (torch.arange(h.shape[1], device=h.device)[None, :]
             < vlen[:, None]).to(h.dtype)
        pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1)[:, None]
        return F.linear(pooled.float(), self.score_w.float()).squeeze(-1)

    @torch.no_grad()
    def score_ids(self, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """[B,S] ids（右 padding）→ [B] logits。B=1 小 S 走图，其余走 eager。"""
        B, S = input_ids.shape
        if B == 1 and attention_mask is None and S <= self.GRAPH_MAX:
            return self._score_graphed(input_ids)
        if attention_mask is None:
            vlen = None
        else:
            vlen = attention_mask.sum(-1).to(torch.int32)
        h = self.forward_hidden(input_ids, vlen)
        return self._pool_score(h, vlen if vlen is not None
                                else torch.full((B,), S, dtype=torch.int32,
                                                device=input_ids.device))

    # ---- CUDA Graph（B=1 小 S；右 padding 到桶位，pad 行嵌入清零 +
    #      自研双向 kernel 的 vlen 掩码，replay 时随 static_vlen 变化）----
    # 桶位上限 2048：自研 kernel 在 2K 以上比 flash 慢 2x+，launch 节省
    # 抵不过注意力变慢；更大的 S 走 eager flash（B=1 无 padding 场景）。
    GRAPH_MAX = 2048

    @staticmethod
    def _graph_bucket(S: int) -> int | None:
        if S > NemotronReranker.GRAPH_MAX:
            return None
        return max(256, (S + 255) // 256 * 256)

    def _capture_graph(self, Sb: int):
        static_ids = torch.zeros(1, Sb, dtype=torch.int64, device="cuda")
        static_vlen = torch.full((1,), Sb, dtype=torch.int32, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):
                self._forward_score(static_ids, static_vlen)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._graph_pool), torch.no_grad():
            static_out = self._forward_score(static_ids, static_vlen)
        if self._graph_pool is None:
            self._graph_pool = g.pool()
        self._graphs[Sb] = (g, static_ids, static_vlen, static_out)

    def _forward_score(self, ids, vlen):
        h = self.forward_hidden(ids, vlen)
        return self._pool_score(h, vlen)

    def _score_graphed(self, ids: torch.Tensor) -> torch.Tensor:
        S = ids.shape[1]
        Sb = self._graph_bucket(S)
        if Sb is None:
            return None
        entry = self._graphs.get(Sb)
        if entry is None:
            self._capture_graph(Sb)
            entry = self._graphs[Sb]
        g, s_ids, s_vlen, s_out = entry
        s_ids[:, :S].copy_(ids)      # 尾部残留 id 无害：嵌入后被 vlen 清零
        s_vlen.fill_(S)
        g.replay()
        return s_out

    # ---- 对外 API ----

    @torch.no_grad()
    def rerank(self, query: str, documents: list[str],
               top_n: int | None = None) -> list[dict]:
        """单 query 对多文档逐对打分（pointwise），返回按 logit 降序。"""
        msgs = [[{"role": "query", "content": [{"type": "text", "text": query}]},
                 {"role": "document", "content": [{"type": "text", "text": d}]}]
                for d in documents]
        scores = []
        for m in msgs:
            enc = self.tok.apply_chat_template(
                m, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt")
            ids = enc["input_ids"].to("cuda")
            if ids.shape[1] > self.max_pos:
                raise ValueError(f"对长 {ids.shape[1]} 超过 {self.max_pos}，需截断")
            scores.append(float(self.score_ids(ids)))
        order = sorted(range(len(documents)), key=lambda i: scores[i],
                       reverse=True)
        n = len(documents) if top_n is None else min(top_n, len(documents))
        return [{"document": documents[i], "relevance_score": scores[i],
                 "index": i} for i in order[:n]]
