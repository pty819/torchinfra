"""jina-reranker-v3.5：Qwen3-0.6B 骨干 + 混合注意力 + projector 余弦分。

骨干权重与 torchinfra 的 Qwen3 同布局。差别只有两处（modeling.py）：
- 按 config.layer_types 在滑窗因果（kv > q - 1024 且因果）和全因果之间切换，
  与 transformers sliding_window_causal_mask_function 同一不等式
- 在 <|embed_token|> / <|rerank_token|> 处取 hidden，过
  Linear(1024→512)-ReLU-Linear(512→512)，文档与查询做余弦相似度

注意力走 SDPA（这张卡上长序列比自研 Triton kernel 快约 2 倍，见 P10）。
norm / QKV / MLP 走已有 Triton 融合。
"""
from pathlib import Path
import json
import os
# hipBLASLt 对这张卡的中等 M（几百到一两千）比默认 hipblas 高约 20%。
# 必须在首次 GEMM 前打开；setdefault 不覆盖用户已经设过的值。
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torchinfra.config import Qwen3Config
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.weights import load_weights
from torchinfra.fp8_ckpt import is_fp8_checkpoint, load_fp8_checkpoint
from torchinfra.fuse import fuse_decode_projections
from torchinfra.triton_kernels import triton_rmsnorm, triton_rmsnorm_amax
from torchinfra.triton_kernels.silu import silu_mul, silu_mul_amax
from torchinfra.fp8_linear import quantize_weight, fp8_linear, fp8_linear_q

@triton.jit
def _window_attn_kernel(
    Q, K, V, O, PART, PAD, sm, S,
    NQ_PER_KV: tl.constexpr, D: tl.constexpr,
    TQ: tl.constexpr, TK: tl.constexpr, WINDOW: tl.constexpr,
):
    """滑窗因果注意力，FA2 式三段循环 + 直写 [B, S, H*D] 输出。

    Q/K/V 布局 [B, H, S, D]（KV 头少 NQ_PER_KV 倍），PAD [B] 为左侧 padding
    长度。输出 O 直接按 [B, S, H*D] 行主序写，滑窗层免去 transpose+contiguous
    拷贝。K 循环按满块条件切三段：窗内侧段只需窗掩码、中段整块免掩码、
    近对角段只需因果掩码——免掩码段省掉 [TQ,TK] 的 where/比较群。
    PART 顺带写每个 [TQ,D] tile 的 amax（bf16 舍入后值，fp8 量化融合用，
    [B, H, cdiv(S,TQ)] fp32；掩码行记 0，不污染全局 amax）。
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    H = tl.num_programs(1)
    # 分段免掩码的前提：远端段因果自动满足、近端段窗自动满足
    tl.static_assert(WINDOW >= TQ + 2 * TK - 2)
    tl.static_assert(WINDOW >= TK)
    kv_h = h // NQ_PER_KV
    pad = tl.load(PAD + b)
    q0 = t * TQ
    offs_q = q0 + tl.arange(0, TQ)
    offs_d = tl.arange(0, D)
    qm = (offs_q < S) & (offs_q >= pad)
    q = tl.load(Q + ((b * H + h) * S + offs_q[:, None]) * D + offs_d[None, :],
                mask=qm[:, None], other=0.0)
    m_i = tl.full([TQ], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([TQ], dtype=tl.float32)
    acc = tl.zeros([TQ, D], dtype=tl.float32)
    start = tl.maximum(q0 - WINDOW + 1, pad)
    end = tl.minimum(q0 + TQ, S)
    # 三段共用锚定在 start 的 TK 网格，杜绝段间块重叠（曾因 clean 段末块
    # 越过界、下一段从界值起步，重叠 k 被双计）。
    clean_lo = tl.maximum(q0 + TQ - WINDOW, start)
    lo_grid = start + tl.cdiv(clean_lo - start, TK) * TK
    # 首个"整块不全因果"的对齐 ks：ks > q0+1-TK ⟺ k_max > q0
    hi_grid = start + tl.cdiv(q0 + 2 - TK - start, TK) * TK
    hi_grid = tl.maximum(hi_grid, lo_grid)

    kv_base = (b * (H // NQ_PER_KV) + kv_h) * S

    # 段 1：远端，只需窗掩码（k_max < q0 ≤ 所有 q，因果自动满足）
    for ks in range(start, lo_grid, TK):
        offs_k = ks + tl.arange(0, TK)
        km = (offs_k < S) & (offs_k >= pad)
        k = tl.load(K + (kv_base + offs_k[:, None]) * D + offs_d[None, :],
                    mask=km[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm
        mask = km[None, :] & (offs_k[None, :] > offs_q[:, None] - WINDOW)
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
    # 段 2：满块免掩码（ks ≥ clean_lo 全在窗内；k_max ≤ q0 全因果）
    for ks in range(lo_grid, hi_grid, TK):
        offs_k = ks + tl.arange(0, TK)
        km = (offs_k < S) & (offs_k >= pad)
        k = tl.load(K + (kv_base + offs_k[:, None]) * D + offs_d[None, :],
                    mask=km[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + (kv_base + offs_k[:, None]) * D + offs_d[None, :],
                    mask=km[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new
    # 段 3：近对角，只需因果掩码（k ≥ lo_grid ≥ clean_lo > q_max - W，窗自动满足）
    for ks in range(hi_grid, end, TK):
        offs_k = ks + tl.arange(0, TK)
        km = (offs_k < S) & (offs_k >= pad)
        k = tl.load(K + (kv_base + offs_k[:, None]) * D + offs_d[None, :],
                    mask=km[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm
        mask = km[None, :] & (offs_k[None, :] <= offs_q[:, None])
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
    tl.store(O + (b * S + offs_q[:, None]) * (H * D) + h * D + offs_d[None, :],
             ob, mask=qm[:, None])
    # tile amax：掩码行（左侧 padding / 末 tile 越界）记 0；l_i=0 的全掩码
    # tile 会得 NaN，必须 where 掉再归约
    obz = tl.where(qm[:, None], ob.to(tl.float32), 0.0)
    nt = tl.cdiv(S, TQ)
    tl.store(PART + (b * H + h) * nt + t, tl.max(tl.abs(obz)))


@triton.jit
def _transpose_amax_kernel(IN, OUT, PART, S, H: tl.constexpr,
                           D: tl.constexpr, BLOCK: tl.constexpr):
    """SDPA 输出 [B, H, S, D] → [B*S, H*D] 行主序 bf16 + 每行 amax partial。

    替代 transpose+contiguous 拷贝与独立 _row_amax：一趟读完，写转置结果
    并顺带吐 fp8 量化要的逐行 amax（bf16 舍入后值，bit-exact）。"""
    bs = tl.program_id(0)
    b = bs // S
    s = bs % S
    offs = tl.arange(0, BLOCK)
    h = offs // D
    d = offs % D
    v = tl.load(IN + ((b * H + h) * S + s) * D + d)
    tl.store(OUT + bs * BLOCK + offs, v)
    tl.store(PART + bs, tl.max(tl.abs(v.to(tl.float32))))


def transpose_amax(out: torch.Tensor):
    """out [B, H, S, D] 连续 → ([B, S, H*D] bf16, partial [B*S] fp32)。"""
    B, H, S, D = out.shape
    flat = torch.empty(B * S, H * D, dtype=out.dtype, device=out.device)
    part = torch.empty(B * S, device=out.device, dtype=torch.float32)
    _transpose_amax_kernel[(B * S,)](out, flat, part, S, H=H, D=D,
                                     BLOCK=H * D, num_warps=8)
    return flat.view(B, S, H * D), part


DOC_ID = 151670
QUERY_ID = 151671
WINDOW = 1024
# 滑窗 kernel 的 launch 配置（applications/nemotron-serving/scripts/bench_window_cfg.py 扫出来的最优档：
# 三段式免掩码下 64/64/w4/st1 最优——st2/大 tile 在 gfx1201 上反而慢 2x）
WIN_TQ = 64
WIN_TK = 64
WIN_WARPS = 4
WIN_STAGES = 1
# qk_norm_rope kernel 的行块/并行度（bench_qk_rope_cfg.py 扫档；TR=8 在
# gfx1201 上有确定性算错的前科，P10 锁 16，动前必须过对拍）
QKR_TR = 16
QKR_WARPS = 8


def _empty_param():
    return torch.nn.Parameter(torch.empty(0, device="cuda", dtype=torch.bfloat16),
                              requires_grad=False)


def _install_saved_fp8(model, fp8: dict) -> None:
    """直接挂 checkpoint 里的 E4M3，q/k/v、gate/up 按存盘时的共用 scale 拼回去。"""
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        p = f"model.layers.{i}.self_attn"
        q, qs = fp8[f"{p}.q_proj.weight"]
        k, ks = fp8[f"{p}.k_proj.weight"]
        v, vs = fp8[f"{p}.v_proj.weight"]
        if not (torch.equal(qs, ks) and torch.equal(qs, vs)):
            raise RuntimeError(f"layer {i} q/k/v scale 不一致，不能拼成一次 GEMM")
        attn.w_qkv_fp8 = torch.cat([q, k, v], dim=0).contiguous()
        attn.w_qkv_s = qs
        attn.o_w_fp8, attn.o_w_s = fp8[f"{p}.o_proj.weight"]
        attn.w_qkv = None
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(attn, name).weight = _empty_param()
        mlp = layer.mlp
        mp = f"model.layers.{i}.mlp"
        g, gs = fp8[f"{mp}.gate_proj.weight"]
        u, us = fp8[f"{mp}.up_proj.weight"]
        if not torch.equal(gs, us):
            raise RuntimeError(f"layer {i} gate/up scale 不一致")
        mlp.w_gu_fp8 = torch.cat([g, u], dim=0).contiguous()
        mlp.w_gu_s = gs
        mlp.down_w_fp8, mlp.down_w_s = fp8[f"{mp}.down_proj.weight"]
        mlp.w_gate_up = None
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(mlp, name).weight = _empty_param()


def _quantize_linears(model) -> None:
    """四个大 GEMM 的权重换成 tensorwise E4M3，丢掉 bf16 副本。"""
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.w_qkv_fp8, attn.w_qkv_s = quantize_weight(attn.w_qkv)
        attn.o_w_fp8, attn.o_w_s = quantize_weight(attn.o_proj.weight.data)
        attn.w_qkv = None
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(attn, name).weight = _empty_param()
        mlp = layer.mlp
        mlp.w_gu_fp8, mlp.w_gu_s = quantize_weight(mlp.w_gate_up)
        mlp.down_w_fp8, mlp.down_w_s = quantize_weight(mlp.down_proj.weight.data)
        mlp.w_gate_up = None
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(mlp, name).weight = _empty_param()


@triton.jit
def _qk_norm_rope_kernel(
    Q, K, V_IN, QNW, KNW, COS, SIN, POS, QO, KO, VO,
    s_q, s_k, s_v, s_cos, eps, n_rows, seq,
    NQ: tl.constexpr, NKV: tl.constexpr,
    D: tl.constexpr, HALF: tl.constexpr, TR: tl.constexpr,
    WANT_V: tl.constexpr,
):
    """一行块 × 一个头：QK-RMSNorm + rotate-half RoPE（可选顺带搬运 v）。

    输入 q/k/v 按 [B*S, H, D] 摊平。POS [B*S] 是各 token 的 RoPE 位置
    （左 padding 的位置从 0 重新数）。输出 q/k 为 [B, H, S, D]；WANT_V 时
    把 v 原样搬到 [B, KV, S, D]，省掉调用方一次 permute+contiguous。
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
    qw1 = tl.load(QNW + off).to(tl.float32)[None, :]
    qw2 = tl.load(QNW + HALF + off).to(tl.float32)[None, :]
    qb = rows[:, None] * s_q + h * D
    a = tl.load(Q + qb + off[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    b = tl.load(Q + qb + HALF + off[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    rms = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1)) / D + eps)
    a = a * rms[:, None] * qw1
    b = b * rms[:, None] * qw2
    db = ((bi[:, None] * NQ + h) * seq + row[:, None]) * D
    tl.store(QO + db + off[None, :], a * cos - b * sin, mask=rm[:, None])
    tl.store(QO + db + HALF + off[None, :], b * cos + a * sin, mask=rm[:, None])
    if h < NKV:
        kw1 = tl.load(KNW + off).to(tl.float32)[None, :]
        kw2 = tl.load(KNW + HALF + off).to(tl.float32)[None, :]
        kb = rows[:, None] * s_k + h * D
        a = tl.load(K + kb + off[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        b = tl.load(K + kb + HALF + off[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        rms = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1)) / D + eps)
        a = a * rms[:, None] * kw1
        b = b * rms[:, None] * kw2
        db = ((bi[:, None] * NKV + h) * seq + row[:, None]) * D
        tl.store(KO + db + off[None, :], a * cos - b * sin, mask=rm[:, None])
        tl.store(KO + db + HALF + off[None, :], b * cos + a * sin, mask=rm[:, None])
        if WANT_V:
            vb = rows[:, None] * s_v + h * D
            for dd in range(0, D, HALF):
                vv = tl.load(V_IN + vb + dd + off[None, :], mask=rm[:, None],
                             other=0.0)
                tl.store(VO + db + dd + off[None, :], vv, mask=rm[:, None])


def _format_prompt(query: str, docs: list[str], instruction: str | None,
                   tok) -> str:
    # 与官方 modeling.format_docs_prompts_func 一致（no_thinking=True）
    for t in ("<|rerank_token|>", "<|embed_token|>"):
        query = query.replace(t, "")
        docs = [d.replace(t, "") for d in docs]
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    body = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    if instruction:
        body += f"<instruct>\n{instruction}\n</instruct>\n"
    body += "\n".join(
        f'<passage id="{i}">\n{doc}<|embed_token|>\n</passage>'
        for i, doc in enumerate(docs))
    body += f"\n<query>\n{query}<|rerank_token|>\n</query>"
    return prefix + body + suffix


class JinaReranker:
    def __init__(self, model: Qwen3ForCausalLM, proj_w1, proj_w2,
                 layer_types: list[str], tok, fp8: bool = False):
        self.model = model
        self.proj_w1 = proj_w1
        self.proj_w2 = proj_w2
        self.layer_types = layer_types
        self.tok = tok
        self.fp8 = fp8
        self._graphs: dict = {}
        self._graph_pool = None

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, dtype=torch.bfloat16,
                        fp8: bool = True):
        model_dir = Path(model_dir)
        cfg = Qwen3Config.from_json(model_dir)
        raw = json.loads((model_dir / "config.json").read_text())
        layer_types = raw["layer_types"]
        assert raw["sliding_window"] == WINDOW
        saved_fp8 = is_fp8_checkpoint(model_dir)
        fp8_tensors = None
        if saved_fp8:
            sd, fp8_tensors = load_fp8_checkpoint(model_dir)
            fp8 = True
        else:
            sd = load_weights(model_dir, dtype=dtype)
        w1 = sd.pop("projector.0.weight")
        w2 = sd.pop("projector.2.weight")
        # plan_blocks 按 tokenizer.model_max_length（131072）切块，RoPE 表必须
        # 覆盖同样范围——曾硬限 32768，>32K 的块会越界读 cos/sin 表直接 GPU
        # 页错误（HSA_STATUS_ERROR_MEMORY_FAULT）。表本身只有 ~134MB。
        rope_max = min(cfg.max_position_embeddings, 131072)
        m = Qwen3ForCausalLM(cfg, max_seq=rope_max,
                             norm_backend="torch").to("cuda").to(dtype)
        from torchinfra.modules import RotaryEmbedding
        m.rope = RotaryEmbedding(cfg.head_dim, m.max_seq, cfg.rope_theta,
                                 device="cuda")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert not missing or set(missing) <= {"lm_head.weight"}
        if cfg.tie_word_embeddings:
            m.lm_head.weight = m.model.embed_tokens.weight
        m.eval()
        fuse_decode_projections(m)
        if fp8_tensors is not None:
            _install_saved_fp8(m, fp8_tensors)
        elif fp8:
            _quantize_linears(m)
        if hasattr(torch.cuda, "tunable"):
            torch.cuda.tunable.enable(True)
            # PYTORCH_TUNABLEOP_TUNING=0 时只用结果文件查表，不为新形状现场调优
            # （新形状退回 hipBLASLt 默认启发式；基准时避免调优吞掉几分钟）。
            if (fp8 and os.environ.get("PYTORCH_TUNABLEOP_TUNING", "1") != "0"
                    and hasattr(torch.cuda.tunable, "tuning_enable")):
                torch.cuda.tunable.tuning_enable(True)
                torch.cuda.tunable.set_max_tuning_iterations(80)
                torch.cuda.tunable.set_max_tuning_duration(15)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        rr = cls(m, w1, w2, layer_types, tok, fp8=fp8)
        rr._graphs = {}
        rr._graph_pool = None
        return rr

    # ---- 整前向 CUDA Graph（小 S 档，launch 开销占前向 ~45%）----
    # 分桶右 padding：因果注意力下真实 token 永远看不到未来位置的 pad，
    # pad 行的输出在打分时被丢弃，因此尾部的陈旧 id 无需清零。
    GRAPH_MAX = 8192

    @staticmethod
    def _graph_bucket(S: int) -> int | None:
        if S > JinaReranker.GRAPH_MAX:
            return None
        if S <= 4096:
            return max(256, (S + 255) // 256 * 256)
        return (S + 511) // 512 * 512

    def _capture_graph(self, Sb: int):
        static_ids = torch.zeros(1, Sb, dtype=torch.int64, device="cuda")
        static_mask = torch.ones(1, Sb, dtype=torch.bool, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):  # 预热：触发 triton JIT 与 GEMM 选型
                self._forward_hidden(static_ids, rowmask=static_mask)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        pool = self._graph_pool
        with torch.cuda.graph(g, pool=pool), torch.no_grad():
            static_out = self._forward_hidden(static_ids, rowmask=static_mask)
        if self._graph_pool is None:
            self._graph_pool = g.pool()
        self._graphs[Sb] = (g, static_ids, static_mask, static_out)

    def _forward_graphed(self, ids: torch.Tensor) -> torch.Tensor | None:
        """B=1 无 padding 时走图；形状超界返回 None 由调用方走 eager。"""
        Sb = self._graph_bucket(ids.shape[1])
        if Sb is None:
            return None
        entry = self._graphs.get(Sb)
        if entry is None:
            self._capture_graph(Sb)
            entry = self._graphs[Sb]
        g, static_ids, static_mask, static_out = entry
        S = ids.shape[1]
        static_ids[:, :S].copy_(ids)
        if S != Sb:
            # 尾部清零标记：pad 行激活归零，不进 tensorwise amax
            static_mask.fill_(True)
            static_mask[:, S:].fill_(False)
        g.replay()
        return static_out[:, :S]

    def _causal_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """valid [B, S] → 加性 mask [B, 1, S, S]。只在有左 padding 时用。"""
        B, S = valid.shape
        idx = torch.arange(S, device=valid.device)
        allow = (valid[:, :, None] & valid[:, None, :]
                 & (idx[None, None, :] <= idx[None, :, None]))
        mask = torch.zeros(B, 1, S, S, dtype=self.proj_w1.dtype, device=valid.device)
        return mask.masked_fill(~allow[:, None], float("-inf"))

    @torch.no_grad()
    def _forward_hidden(self, input_ids: torch.Tensor,
                        valid: torch.Tensor | None = None,
                        padded: bool | None = None,
                        rowmask: torch.Tensor | None = None
                        ) -> torch.Tensor:
        """input_ids [B, S] → hidden [B, S, E]。valid 为 True 的位置是真实 token（左 padding）。

        padded 由调用方在 CPU 侧判定传入（tokenizer 的 attention_mask 在 CPU
        上就有）；None 时按 valid 是否提供推断——自己构造的全 1 视为无 padding。
        不得在此处 bool(valid.all())：那是主机同步，eager 期挡流水、
        CUDA Graph 捕获期直接作废捕获。

        rowmask [B, S] bool（图路径专用）：嵌入后把 False 行清零。右 padding
        行若不清零，其激活会进入 tensorwise fp8 量化的 amax，改变整批 scale，
        使图路径与 eager 结果不一致；清零后逐行算子下零行恒为零，amax 不受影响。
        """
        model = self.model
        if input_ids.shape[1] > model.max_seq:
            raise ValueError(
                f"序列长度 {input_ids.shape[1]} 超过 RoPE 表容量 {model.max_seq}；"
                "plan_blocks 应保证不超限")
        x = model.model.embed_tokens(input_ids)
        if rowmask is not None:
            x = x * rowmask.unsqueeze(-1).to(x.dtype)
        B, S, _ = x.shape
        if valid is None:
            valid = torch.ones(B, S, dtype=torch.bool, device=x.device)
            padded = False
        elif padded is None:
            padded = not bool(valid.all().cpu())
        pos = (valid.long().cumsum(-1) - 1).clamp(min=0).to(torch.int32)
        pad = (~valid).sum(-1).to(torch.int32)
        causal_mask = self._causal_mask(valid) if padded else None
        cos, sin = model.rope.cos, model.rope.sin
        half = cos.shape[1] // 2
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            eps = attn.q_norm.eps
            if self.fp8:
                # 量化融合：rmsnorm/silu 顺带吐每行 amax partial，fp8_linear_q
                # 跳过全量 _row_amax 读（每层省 3 kernel + 3 趟激活读，数值
                # 与融合前 bit-exact：partial 用 bf16 舍入后的值算）
                h, ph = triton_rmsnorm_amax(x, layer.input_layernorm.weight,
                                            eps)
                qkv = fp8_linear_q(h, attn.w_qkv_fp8, attn.w_qkv_s, ph)
            else:
                h = triton_rmsnorm(x, layer.input_layernorm.weight, eps)
                qkv = F.linear(h, attn.w_qkv)
            nq = attn.n_heads * attn.hd
            nk = attn.n_kv * attn.hd
            q_in = qkv[..., :nq].reshape(B * S, attn.n_heads, attn.hd)
            k_in = qkv[..., nq:nq + nk].reshape(B * S, attn.n_kv, attn.hd)
            v_in = qkv[..., nq + nk:].reshape(B * S, attn.n_kv, attn.hd)
            q = torch.empty(B, attn.n_heads, S, attn.hd, dtype=q_in.dtype,
                            device=q_in.device)
            k = torch.empty(B, attn.n_kv, S, attn.hd, dtype=q_in.dtype,
                            device=q_in.device)
            v = torch.empty(B, attn.n_kv, S, attn.hd, dtype=q_in.dtype,
                            device=q_in.device)
            _qk_norm_rope_kernel[(triton.cdiv(B * S, QKR_TR), attn.n_heads)](
                q_in, k_in, v_in, attn.q_norm.weight, attn.k_norm.weight,
                cos, sin, pos.reshape(-1), q, k, v,
                q_in.stride(0), k_in.stride(0), v_in.stride(0), cos.stride(0),
                eps, B * S, S,
                NQ=attn.n_heads, NKV=attn.n_kv, D=attn.hd, HALF=half,
                TR=QKR_TR, WANT_V=True, num_warps=QKR_WARPS)
            # S<=WINDOW 时滑窗约束被因果完全覆盖，走 is_causal 的快 kernel
            sliding = self.layer_types[i] == "sliding_attention" and S > WINDOW
            if sliding:
                out = torch.empty(B, S, nq, dtype=q.dtype, device=q.device)
                nt = triton.cdiv(S, WIN_TQ)
                apart = torch.empty(B, attn.n_heads, nt, dtype=torch.float32,
                                    device=q.device)
                _window_attn_kernel[(B, attn.n_heads, nt)](
                    q, k, v, out, apart, pad, attn.hd ** -0.5, S,
                    NQ_PER_KV=attn.n_heads // attn.n_kv, D=attn.hd,
                    TQ=WIN_TQ, TK=WIN_TK, WINDOW=WINDOW,
                    num_warps=WIN_WARPS, num_stages=WIN_STAGES)
                attn_out = out  # 已是 [B, S, H*D] 行主序
                attn_partial = apart
            elif padded:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=causal_mask, enable_gqa=True)
                attn_out, attn_partial = transpose_amax(out)
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, enable_gqa=True)
                attn_out, attn_partial = transpose_amax(out)
            if self.fp8:
                x = x + fp8_linear_q(attn_out, attn.o_w_fp8, attn.o_w_s,
                                     attn_partial)
                h2, ph2 = triton_rmsnorm_amax(
                    x, layer.post_attention_layernorm.weight, eps)
                gu = fp8_linear_q(h2, layer.mlp.w_gu_fp8, layer.mlp.w_gu_s, ph2)
                act, pa = silu_mul_amax(gu.reshape(B * S, -1))
                x = x + fp8_linear_q(act, layer.mlp.down_w_fp8,
                                     layer.mlp.down_w_s, pa).view(B, S, -1)
            else:
                x = x + attn.o_proj(attn_out)
                h2 = triton_rmsnorm(x, layer.post_attention_layernorm.weight, eps)
                act = silu_mul(F.linear(h2, layer.mlp.w_gate_up).reshape(B * S, -1))
                x = x + F.linear(act, layer.mlp.down_proj.weight).view(B, S, -1)
        return triton_rmsnorm(x, model.model.norm.weight, model.model.norm.eps)

    def _project(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(F.relu(F.linear(h, self.proj_w1)), self.proj_w2)

    @torch.no_grad()
    def score_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """单条 prompt → 各文档余弦分 [n_docs]。"""
        doc, qry = self._embed_ids(input_ids)
        return F.cosine_similarity(doc, qry.expand_as(doc), dim=-1)

    @torch.no_grad()
    def _embed_ids(self, input_ids: torch.Tensor):
        """单条 prompt → (doc [n, 512], query [1, 512])，float32。"""
        rows = self.embed_batch(input_ids)
        return rows[0]

    @torch.no_grad()
    def embed_batch(self, input_ids: torch.Tensor,
                    valid: torch.Tensor | None = None, padded: bool | None = None
                    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """一批 prompt（左 padding）→ 每条 (doc [n_i, 512], query [1, 512])。"""
        if valid is None and input_ids.shape[0] == 1:
            h = self._forward_graphed(input_ids)
            if h is None:
                h = self._forward_hidden(input_ids, valid, padded)
        else:
            h = self._forward_hidden(input_ids, valid, padded)
        out = []
        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            doc = self._project(h[b, ids == DOC_ID]).float()
            qry = self._project(h[b, ids == QUERY_ID]).float()
            out.append((doc, qry))
        return out

    def _truncate_texts(self, query: str, documents: list[str],
                        max_query_length: int, max_doc_length: int):
        """与官方 _truncate_texts 相同：超长才 decode 回文本，长度按截断后的 token 数。"""
        docs, doc_lengths = [], []
        for doc in documents:
            ids = self.tok(doc, truncation=True, max_length=max_doc_length)["input_ids"]
            if len(ids) >= max_doc_length:
                doc = self.tok.decode(ids)
            docs.append(doc)
            doc_lengths.append(len(ids))
        qids = self.tok(query, truncation=True, max_length=max_query_length)["input_ids"]
        if len(qids) >= max_query_length:
            query = self.tok.decode(qids)
        return query, docs, doc_lengths, len(qids)

    def plan_blocks(self, query: str, documents: list[str]
                    ) -> tuple[str, list[list[str]]]:
        """切成若干块。规则与官方 rerank 相同：满 125 篇，或剩余预算放不下一篇最长文档。"""
        max_length = self.tok.model_max_length
        max_query_length, max_doc_length, block_size = 1024, 8192, 125
        query, docs, doc_lengths, query_length = self._truncate_texts(
            query, documents, max_query_length, max_doc_length)
        length_capacity = max_length - 2 * query_length
        blocks: list[list[str]] = []
        block: list[str] = []
        for length, doc in zip(doc_lengths, docs):
            block.append(doc)
            length_capacity -= length
            if len(block) >= block_size or length_capacity <= max_doc_length:
                blocks.append(block)
                block = []
                length_capacity = max_length - 2 * query_length
        if block:
            blocks.append(block)
        return query, blocks

    def combine(self, doc_embeds: list[torch.Tensor],
                query_embeds: list[torch.Tensor], weights: list[float],
                documents: list[str], top_n: int | None,
                return_embeddings: bool) -> list[dict]:
        """各块的 query 向量按块内最高分加权平均，再和全部文档向量做余弦。"""
        w = torch.tensor(weights, dtype=torch.float32,
                         device=query_embeds[0].device)
        q = (torch.cat(query_embeds, 0) * w[:, None]).sum(0) / w.sum()
        docs_h = torch.cat(doc_embeds, 0)
        scores = F.cosine_similarity(docs_h, q.expand_as(docs_h), dim=-1).cpu()
        order = torch.argsort(scores, descending=True).tolist()
        n = len(documents) if top_n is None else min(top_n, len(documents))
        return [{
            "document": documents[i],
            "relevance_score": float(scores[i]),
            "index": i,
            "embedding": docs_h[i].cpu() if return_embeddings else None,
        } for i in order[:n]]

    @torch.no_grad()
    def rerank(self, query: str, documents: list[str],
               top_n: int | None = None,
               return_embeddings: bool = False) -> list[dict]:
        """单请求重排。多块时逐块前向；并发请求走 RerankScheduler，把块拼进同一个 batch。"""
        if not documents:
            return []
        query, blocks = self.plan_blocks(query, documents)
        doc_embeds, query_embeds, weights = [], [], []
        for block in blocks:
            ids = self.tok(_format_prompt(query, block, None, self.tok),
                           return_tensors="pt").input_ids.to("cuda")
            doc, qry = self._embed_ids(ids)
            scores = F.cosine_similarity(doc, qry.expand_as(doc), dim=-1)
            doc_embeds.append(doc)
            query_embeds.append(qry)
            weights.append(float(((1.0 + scores) / 2.0).max()))
        return self.combine(doc_embeds, query_embeds, weights, documents,
                            top_n, return_embeddings)
