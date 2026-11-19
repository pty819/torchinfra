#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""unsloth fp8 checkpoint → QwenDiT 加载器（键名映射 + bf16/fp8 双模式）"""
import torch
from safetensors.torch import load_file
from dit_forward import QwenDiT, FP8OrBF16Linear, N_LAYERS

CKPT = "~/models/qwen21/fp8/qwen21_fp8_e4m3fn.safetensors"

# checkpoint 键 → 模型属性路径
MAP = [
    ("txt_in.text_norm.weight",            "txt_norm.weight",           "bf16"),
    ("txt_in.in_layer",                    "txt_in_layer",              "fp8pair"),
    ("txt_in.out_layer",                   "txt_out_layer",             "fp8pair"),
    ("img_in.weight",                      "img_in.weight",             "bf16"),
    ("time_text_embed.timestep_embedder.linear_1.weight", "t_embed_1.weight", "bf16"),
    ("time_text_embed.timestep_embedder.linear_2", "t_embed_2",          "fp8pair"),
    ("modulation.1",                       "modulation",                "fp8pair"),
    ("norm_out.linear",                    "norm_out_linear",           "fp8pair"),
    ("proj_out.weight",                    "proj_out.weight",           "bf16"),
]


def load_fp8(path=CKPT, device="cpu", mode="bf16", num_layers=N_LAYERS):
    sd = load_file(path)
    model = QwenDiT(num_layers=num_layers).to(torch.bfloat16)

    def get_pair(prefix):
        return sd[prefix + "._weight_qdata"], sd[prefix + "._weight_scale"]

    for ck, attr, kind in MAP:
        if kind == "bf16":
            tgt = model.get_submodule(attr)
            tgt.weight.data.copy_(sd[ck].to(torch.bfloat16))
        else:
            q, s = get_pair(ck)
            out_f, in_f = q.shape
            lin = FP8OrBF16Linear(q, s, mode=mode).to(torch.bfloat16)
            setattr(model, attr, lin)

    for n in range(num_layers):
        blk = model.blocks[n]
        p = f"transformer_blocks.{n}"
        pairs = [(f"{p}.attn.to_q", "attn_to_q"), (f"{p}.attn.to_k", "attn_to_k"),
                 (f"{p}.attn.to_v", "attn_to_v"), (f"{p}.attn.to_out.0", "attn_to_out"),
                 (f"{p}.img_mlp.gate_layer", "mlp_gate"), (f"{p}.img_mlp.proj", "mlp_up"),
                 (f"{p}.img_mlp.out", "mlp_down")]
        for ck, attr in pairs:
            q, s = get_pair(ck)
            setattr(blk, attr, FP8OrBF16Linear(q, s, mode=mode).to(torch.bfloat16))
        blk.attn_norm_q.weight.data.copy_(sd[f"{p}.attn.norm_q.weight"].to(torch.bfloat16))
        blk.attn_norm_k.weight.data.copy_(sd[f"{p}.attn.norm_k.weight"].to(torch.bfloat16))

    return model.to(device).eval()


if __name__ == "__main__":
    m = load_fp8(device="cpu", mode="bf16")
    n_lin = sum(1 for x in m.modules() if isinstance(x, FP8OrBF16Linear))
    print(f"加载成功: fp8linear={n_lin} (期望 7+32*7=231), params≈{sum(p.numel() for p in m.parameters())/1e9:.2f}B")
