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
