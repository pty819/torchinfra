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
