"""Qwen3 整模型：HF 参数名 + from_pretrained。"""
from pathlib import Path
import torch
import torch.nn as nn
from torchinfra.config import Qwen3Config
from torchinfra.modules import RMSNorm, RotaryEmbedding, DecoderLayer
from torchinfra.kv_cache import KVCache
from torchinfra.weights import load_weights


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config, norm_backend: str = "torch"):
        super().__init__()
        if norm_backend == "triton":
            from torchinfra.triton_kernels import TritonRMSNorm
            norm_cls = TritonRMSNorm
        else:
            norm_cls = RMSNorm
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, norm_backend)
            for _ in range(cfg.num_hidden_layers))
        self.norm = norm_cls(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids, rope, kv, start_pos):
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, rope, kv, i, start_pos)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config, max_seq: int = 8192,
                 norm_backend: str = "torch"):
        super().__init__()
        self.cfg = cfg
        self.max_seq = max_seq
        self.model = Qwen3Model(cfg, norm_backend)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope = None  # from_pretrained 里在 dtype cast 之后创建（保 fp32）

    def forward(self, input_ids, kv: KVCache, start_pos: int = 0,
                logits_to_keep: str = "last"):
        h = self.model(input_ids, self.rope, kv, start_pos)
        h = h[:, -1:] if logits_to_keep == "last" else h
        return self.lm_head(h)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device: str = "cuda",
                        dtype=torch.bfloat16, max_seq: int = 8192,
                        backend: str = "torch", compile: bool = False):
        cfg = Qwen3Config.from_json(model_dir)
        m = cls(cfg, max_seq=max_seq,
                norm_backend=backend).to(device=device).to(dtype)
        # rope 的 cos/sin 必须保 fp32（大位置角对精度敏感），所以在 cast 之后创建
        if backend == "triton":
            from torchinfra.triton_kernels import TritonRoPE
            m.rope = TritonRoPE(cfg.head_dim, max_seq, cfg.rope_theta,
                                device=device)
        else:
            m.rope = RotaryEmbedding(cfg.head_dim, max_seq, cfg.rope_theta,
                                     device=device)
        sd = load_weights(model_dir, device=device, dtype=dtype)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, f"多余权重: {unexpected}"
        # tied 时 checkpoint 可能缺 lm_head（missing 允许）也可能两份都存
        assert not missing or (cfg.tie_word_embeddings
                               and set(missing) <= {"lm_head.weight"}), \
            f"权重缺失: {missing}"
        if cfg.tie_word_embeddings:
            m.lm_head.weight = m.model.embed_tokens.weight  # 真正共享存储
        m.eval()
        if compile:
            # 只加速 prefill 路径（实测 1.68x @512token）；decode_step 直接调
            # 子模块不受影响（实测 compile 于 decode 反而慢 0.74x，graph break
            # 在逐请求 cache append 的 int() 同步点上）
            return _CompiledWrap(m)
        return m


class _CompiledWrap:
    """__call__ 走 torch.compile，其余属性委托原模型。

    不直接改 m.forward：torch.compile 会回调原模块 forward，实例属性替换
    会造成无限递归。
    """

    def __init__(self, m: "Qwen3ForCausalLM"):
        self._m = m
        self._c = torch.compile(m)

    def __call__(self, *args, **kwargs):
        return self._c(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._m, name)
