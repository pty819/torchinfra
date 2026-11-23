#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""unsloth fp8 checkpoint → QwenDiT 加载器（meta 骨架零内存装配, 避免 OOM）"""
from .paths import MODEL_DIR
import os
import torch
import torch.nn as nn
from safetensors.torch import load_file
from .dit_forward import QwenDiT, FP8OrBF16Linear, N_LAYERS

CKPT = str(MODEL_DIR / "fp8/qwen21_fp8_e4m3fn.safetensors")

MAP = [
    ("txt_in.text_norm.weight", "txt_norm", "norm"),
    ("txt_in.in_layer", "txt_in_layer", "fp8"),
    ("txt_in.out_layer", "txt_out_layer", "fp8"),
    ("img_in.weight", "img_in", "lin"),
    ("time_text_embed.timestep_embedder.linear_1.weight", "t_embed_1", "lin"),
    ("time_text_embed.timestep_embedder.linear_2", "t_embed_2", "fp8"),
    ("modulation.1", "modulation", "fp8"),
    ("norm_out.linear", "norm_out_linear", "fp8"),
    ("proj_out.weight", "proj_out", "lin"),
]


def load_fp8(path=CKPT, device="cuda", mode="fp8", num_layers=N_LAYERS):
    with torch.device("meta"):
        model = QwenDiT(num_layers=num_layers)
    sd = load_file(path)  # CPU RAM ~7.1GB, 装配完释放

    def put_param(attr, w):
        mod = model.get_submodule(attr)  # attr 指向模块, 换它的 .weight
        mod.weight = nn.Parameter(w, requires_grad=False)

    for ck, attr, kind in MAP:
        if kind == "norm":
            put_param(attr, sd[ck].to(device=device, dtype=torch.bfloat16))
        elif kind == "lin":
            put_param(attr, sd[ck].to(device=device, dtype=torch.bfloat16))
        else:
            q = sd[ck + "._weight_qdata"].to(device)
            s = sd[ck + "._weight_scale"].to(device)
            setattr(model, attr, FP8OrBF16Linear(q, s, mode=mode))

    for n in range(num_layers):
        blk = model.blocks[n]
        p = f"transformer_blocks.{n}"
        put_param(f"blocks.{n}.attn_norm_q", sd[f"{p}.attn.norm_q.weight"].to(device=device, dtype=torch.bfloat16))
        put_param(f"blocks.{n}.attn_norm_k", sd[f"{p}.attn.norm_k.weight"].to(device=device, dtype=torch.bfloat16))
        pairs = [(f"{p}.attn.to_q", "attn_to_q"), (f"{p}.attn.to_k", "attn_to_k"),
                 (f"{p}.attn.to_v", "attn_to_v"), (f"{p}.attn.to_out.0", "attn_to_out"),
                 (f"{p}.img_mlp.gate_layer", "mlp_gate"), (f"{p}.img_mlp.proj", "mlp_up"),
                 (f"{p}.img_mlp.out", "mlp_down")]
        for ck, attr in pairs:
            q = sd[ck + "._weight_qdata"].to(device)
            s = sd[ck + "._weight_scale"].to(device)
            setattr(blk, attr, FP8OrBF16Linear(q, s, mode=mode))

    del sd
    if os.environ.get("QDIT_LORA", "1") == "1":
        # 风评最佳 NSFW 固定套装 (qdit 模式唯一配置, 不暴露开关)
        from .lora import apply_lora
        LORA_DIR = str(MODEL_DIR / "loras")
        for fname, s_ in (("nsfw_alpacas_v2", 1.0), ("nsfw_vaginus_v13b", 1.0), ("nsfw_genitalia_fix", 1.0)):
            n = apply_lora(model, f"{LORA_DIR}/{fname}.safetensors", s_, device=device)
            print(f"[lora] {fname}@{s_}: {n} 目标已挂")
    return model.eval()


if __name__ == "__main__":
    m = load_fp8(device="cuda", mode="fp8")
    n_lin = sum(1 for x in m.modules() if isinstance(x, FP8OrBF16Linear))
    print(f"加载成功: fp8linear={n_lin} (期望 7+32*7=231) | 显存 {torch.cuda.memory_allocated()/1e9:.2f}GB")
