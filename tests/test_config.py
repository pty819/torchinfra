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
