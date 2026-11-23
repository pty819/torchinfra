#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PEFT LoRA 运行时叠加（alpacas/omni/vaginus 等）
关键: alpacas 系 LoRA 打的是融合名 img_mlp.gate_up，而 unsloth fp8 底是分离布局 →
     lora_B 行 [0:12288]→gate_layer, [12288:24576]→up(proj)。gate 在前（ComfyUI 约定）。
"""
import re
import torch
import torch.nn as nn
from safetensors.torch import load_file

INTER = 12288
BLOCK_MAP = {
    "attn.to_q": "attn_to_q", "attn.to_k": "attn_to_k", "attn.to_v": "attn_to_v",
    "attn.to_out.0": "attn_to_out",
    "img_mlp.gate_layer": "mlp_gate", "img_mlp.proj": "mlp_up", "img_mlp.out": "mlp_down",
}
TOP_MAP = {"txt_in.in_layer": "txt_in_layer", "txt_in.out_layer": "txt_out_layer",
           "img_in": "img_in", "modulation.1": "modulation"}


class LoRAWrapper(nn.Module):
    def __init__(self, base, A, B, mult):
        super().__init__()
        self.base, self.mult = base, mult
        self.A = A.detach().clone()  # [r, in]
        self.B = B.detach().clone()  # [out, r]

    def forward(self, x):
        out = self.base(x)
        delta = (x.float() @ self.A.t().float()) @ self.B.t().float() * self.mult
        return out + delta.to(out.dtype)


def _targets(mod_path):
    """checkpoint 模块路径 → [(模型属性路径, lora_B 行切片)]；gate_up 拆两份"""
    mod_path = mod_path.removeprefix("diffusion_model.")
    m = re.match(r"transformer_blocks\.(\d+)\.(.+)", mod_path)
    if m:
        n, rest = m.groups()
        if rest == "img_mlp.gate_up":
            return [(f"blocks.{n}.mlp_gate", slice(0, INTER)),
                    (f"blocks.{n}.mlp_up", slice(INTER, 2 * INTER))]
        return [(f"blocks.{n}.{BLOCK_MAP[rest]}", None)]
    return [(TOP_MAP[mod_path], None)]


def apply_lora(model, path, strength=1.0, device="cpu", dtype=torch.bfloat16):
    sd = load_file(path)
    pairs, applied = {}, 0
    for k, v in sd.items():
        m = re.match(r"(.+)\.lora_(A|B)\.weight$", k)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = v
    for mod, ab in pairs.items():
        A = ab["A"].to(device=device, dtype=dtype)
        B = ab["B"].to(device=device, dtype=dtype)
        for attr, sl in _targets(mod):
            b = B if sl is None else B[sl].contiguous()
            base = model.get_submodule(attr)
            if isinstance(base, LoRAWrapper):
                base = base.base  # 叠多个 LoRA 时链式包
            wrapper = LoRAWrapper(base, A, b, strength)
            parent, _, name = attr.rpartition(".")
            setattr(model.get_submodule(parent), name, wrapper)   # 注意: 不能对 wrapper 整体 .to(dtype) — 会把 fp8 基底 qdata 转 bf16 毁掉
            applied += 1
    return applied
