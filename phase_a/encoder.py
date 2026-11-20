# -*- coding: utf-8 -*-
"""Qwen3-VL-8B 编码器 CPU 加载与编码 (fp8 权重手动反量化, 零额外依赖)
配方照抄 diffusers QwenImage21Pipeline: 模板 / 丢 system token / pre-norm hidden"""
import json, torch
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

ENC_DIR = "~/models/qwen21/fp8/qwen3vl8b_fp8"
TPL = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
SYS_TXT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"

def load_encoder(keep_lm_head=False):
    """返回 (tok, enc, lm, drop)。enc 留在 CPU/bf16; visual 塔与 lm_head 留 meta 省内存"""
    tok = AutoTokenizer.from_pretrained(ENC_DIR)
    cfg = json.load(open(f"{ENC_DIR}/config.json")); cfg.pop("quantization_config", None)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg))
    torch.set_default_dtype(old)
    from safetensors.torch import safe_open
    sd = {}
    with safe_open(f"{ENC_DIR}/model.safetensors", framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".weight_scale") or k.startswith("model.visual."):
                continue
            if k.startswith("lm_head.") and not keep_lm_head:
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                t = (t.to(torch.float32) * f.get_tensor(k + "_scale")).to(torch.bfloat16)
            sd[k] = t.to(torch.bfloat16)
    enc.load_state_dict(sd, strict=False, assign=True)
    del sd
    for m in enc.modules():  # meta 初始化留下的 inv_freq 需重算, 全零会毁 rope
        inv = getattr(m, "inv_freq", None)
        if inv is not None and inv.is_meta:
            dim = inv.shape[-1] * 2
            m.inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    enc = enc.eval()
    drop = tok(SYS_TXT, return_tensors="pt")["input_ids"].shape[1]
    return tok, enc, enc.model.language_model, drop

@torch.no_grad()
def encode_raw(enc, tok, text, use_autocast=False, runner=None):
    """整段模板编码, 返回 (pre-norm hidden.float(), n_tokens)。hook 由调用方管理"""
    text = text if text else " "
    ids = tok(TPL.format(text), return_tensors="pt")["input_ids"]
    fwd = runner or enc
    if use_autocast:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = fwd(ids, output_hidden_states=True, use_cache=False)
    else:
        out = fwd(ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0].float(), ids.shape[1]
