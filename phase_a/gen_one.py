#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qdit 首张真图: Qwen3-VL(CPU,fp8权重) 编码 → DiT fp8(GPU) 采样 → VAE(GPU) 解码
条件配方照抄 diffusers QwenImage21Pipeline: 模板/丢system token/pre-norm hidden/负向词" "
"""
import sys, time, torch
import torch.nn.functional as F

DEV = "cuda"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "a lovely cat holding a wooden sign that says 'qwen fp8'"
NEG = sys.argv[2] if len(sys.argv) > 2 else ""
W = H = 1344
STEPS = 20

# ---------- 1. 文本编码 (CPU) ----------
t0 = time.time()
import json
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration
ENC_DIR = "~/models/qwen21/fp8/qwen3vl8b_fp8"
tok = AutoTokenizer.from_pretrained(ENC_DIR)
# rocm10 环境无 compressed-tensors/torchao → meta 骨架 + 手动反量化 (F8 .weight × .weight_scale → bf16)
_cfg = json.load(open(f"{ENC_DIR}/config.json")); _cfg.pop("quantization_config", None)
_old_dt = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
with torch.device("meta"):
    enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**_cfg))
torch.set_default_dtype(_old_dt)
from safetensors.torch import safe_open
_sd = {}
with safe_open(f"{ENC_DIR}/model.safetensors", framework="pt", device="cpu") as _f:
    for _k in _f.keys():
        if _k.endswith(".weight_scale") or _k.startswith("model.visual.") or _k.startswith("lm_head."):
            continue  # scale 会并进 weight; visual/lm_head 文本-only 用不到, 留 meta 省内存
        _t = _f.get_tensor(_k)
        if _t.dtype == torch.float8_e4m3fn:
            _t = (_t.to(torch.float32) * _f.get_tensor(_k + "_scale")).to(torch.bfloat16)
        _sd[_k] = _t.to(torch.bfloat16)
enc.load_state_dict(_sd, strict=False, assign=True)
del _sd
# 修 meta 初始化留下的 inv_freq buffer (rotary 需要, 全零会毁 rope)
for _m in enc.modules():
    _inv = getattr(_m, "inv_freq", None)
    if _inv is not None and _inv.is_meta:
        _dim = _inv.shape[-1] * 2
        _m.inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, _dim, 2, dtype=torch.float32) / _dim))
enc = enc.eval()
lm = enc.model.language_model
# 关键: 取最终 RMSNorm 之前的 hidden (transformers 5.x 会给 norm 后的)
handle = lm.norm.register_forward_hook(lambda m, a, o: a[0])
TPL = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

@torch.no_grad()
def encode(text):
    text = text if text else " "
    ids = tok(TPL.format(text), return_tensors="pt")
    out = enc(ids["input_ids"], output_hidden_states=True)
    h = out.hidden_states[-1][0].float()
    sys_len = tok(TPL.split("user")[0] + "user\n", return_tensors="pt")["input_ids"].shape[1]
    return h[:0] if False else h  # 先不丢, 下面统一处理

# 精确计算 system 段 token 数 (照管线: 对 system 消息单独套 chat template)
sys_txt = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
drop = tok(sys_txt, return_tensors="pt")["input_ids"].shape[1]

@torch.no_grad()
def encode2(text):
    text = text if text else " "
    ids = tok(TPL.format(text), return_tensors="pt")
    out = enc(ids["input_ids"], output_hidden_states=True)
    h = out.hidden_states[-1][0].float()
    return h[drop:]
ctx = encode2(PROMPT)
neg = encode2(NEG)
handle.remove()
del enc
print(f"[{time.time()-t0:.0f}s] 编码完成: ctx {tuple(ctx.shape)}, neg {tuple(neg.shape)}, drop={drop}")
if neg.shape[0] != ctx.shape[0]:  # 负向词补齐/截到同长 (CFG 双批需要)
    if neg.shape[0] < ctx.shape[0]:
        neg = torch.cat([neg, neg.new_zeros(ctx.shape[0]-neg.shape[0], neg.shape[1])])
    neg = neg[: ctx.shape[0]]
ctx_d = torch.stack([ctx, neg]).to(DEV).to(torch.bfloat16)

# ---------- 2. DiT 采样 (GPU, eager) ----------
from loader import load_fp8
from sample import sample
from sigmas import flux_sigmas
model = load_fp8(device=DEV, mode="fp8")
Hl = Wl = H // 16
n_img = Hl * Wl
sig = flux_sigmas(STEPS, n_img).tolist()
t0 = time.time()

@torch.no_grad()
def model_fn(x, c, t, pe, prefix):
    return model(x, c, t, pe, prefix)

x = sample(model_fn, ctx_d[0:1], ctx_d[1:2], (64, Hl, Wl), sig, cfg=4.0, seed=42, device=DEV)
print(f"[{time.time()-t0:.0f}s] 采样完成: {tuple(x.shape)}", flush=True)
del model, model_fn; import gc; gc.collect(); torch.cuda.empty_cache()  # 给 VAE 腾显存

# ---------- 3. VAE 解码 (GPU) ----------
t0 = time.time()
import json
from diffusers.models import AutoencoderKLQwenImage21
cfgd = json.load(open("~/models/qwen21/fp8/vae21_config.json"))
cfgd.pop("_class_name", None); cfgd.pop("_diffusers_version", None)
vae = AutoencoderKLQwenImage21(**cfgd).to(DEV, torch.bfloat16).eval()
vae.enable_tiling()  # 大图解码省显存
from safetensors.torch import load_file
raw = load_file("~/models/qwen21/qwen_image_2.1_vae_diffusers.safetensors")
sd = vae.state_dict()
remap = {}
model_keys = set(sd.keys())
for k, v in raw.items():
    cands = [k, "encoder." + k] if not k.startswith(("decoder", "encoder")) else [k]
    for c in cands:
        if c in model_keys and sd[c].shape == v.shape:
            remap[c] = v; break
missing = [k for k in model_keys if k not in remap]
print(f"VAE 映射: {len(remap)}/{len(model_keys)} 命中, 缺 {len(missing)}")
if missing:
    print("  缺样例:", missing[:5]); print("  原始键样例:", list(raw.keys())[:5])
missing2 = [k for k in missing if "quant" not in k]
assert not missing2, f"VAE 键位对不齐: {missing2[:5]}"
vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in remap.items()}, strict=False)
mean = torch.tensor(cfgd["latents_mean"], device=DEV).view(1, 64, 1, 1, 1)
std = torch.tensor(cfgd["latents_std"], device=DEV).view(1, 64, 1, 1, 1)
lat = (x.to(DEV, torch.bfloat16).unsqueeze(2) * std + mean).to(torch.bfloat16)  # 先插帧维 [B,64,1,Hl,Wl] 再乘
with torch.no_grad():
    img = vae.decode(lat, return_dict=False)[0][:, :, 0]      # 取第 0 帧
img = ((img.float() / 2 + 0.5).clamp(0, 1) * 255).byte()[0].cpu()
from PIL import Image
Image.fromarray(img.permute(1, 2, 0).numpy()).save("~/Pictures/qwen/qdit_first.png")
print(f"[{time.time()-t0:.0f}s] 已存 ~/Pictures/qwen/qdit_first.png")
