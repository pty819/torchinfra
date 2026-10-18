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
    # 该 checkpoint 虽然 tied，但文件里同时存了 lm_head（值与 embed 相同）
    if "lm_head.weight" in sd:
        assert sd["lm_head.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
        assert torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"])
    print(f"weights OK: {len(sd)} 个张量")


if __name__ == "__main__":
    main()
