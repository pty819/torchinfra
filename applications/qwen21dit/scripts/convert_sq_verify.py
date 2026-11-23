# SmoothQuant 转换 + W8A8 (reduce_range=False, VNNI) 验证, 扫 α
from qwen21dit.paths import MODEL_DIR
import json, time, sys, gc, torch, os
import torch.nn as nn
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from safetensors.torch import safe_open, save_file
from qwen21dit.encoder import load_encoder, TPL

ENC_DIR = str(MODEL_DIR / "fp8/qwen3vl8b_fp8")
CAL = str(MODEL_DIR / "int8_qwen3vl8b.calib.safetensors")
PROMPTS = ["masterpiece, a serene mountain lake at dawn, mist over water, ultra detailed",
           "1girl, cute anime style, cherry blossoms, golden hour lighting, ultra detailed",
           " masterpiece, best quality, 1girl, seductive pose, dim lighting, detailed skin"]
ALPHAS = [0.75]
MIXED = os.environ.get("SQ_MIXED", "1") == "1"
RATIO_TH = 32.0

tok = AutoTokenizer.from_pretrained(ENC_DIR)

# ---- bf16 基线 ----
tok_, enc_bf, _, _ = load_encoder()
base = {}
with torch.no_grad():
    for p in PROMPTS:
        ids = tok_(TPL.format(p), return_tensors="pt")["input_ids"]
        out = enc_bf(ids, output_hidden_states=True, use_cache=False)
        base[p] = out.hidden_states[-1][0].float().cpu()
del tok_, enc_bf, out
gc.collect()

cal = {}
with safe_open(CAL, framework="pt") as f:
    for k in f.keys():
        cal[k] = f.get_tensor(k)

import torch.ao.nn.quantized.dynamic as qdyn

class W16Linear(torch.nn.Module):
    """问题层: W8A16 流式反量化(带 smooth, 权重已含 s)"""
    def __init__(self, wq, s, bias):
        super().__init__()
        self.register_buffer("sinv", (1.0 / s).float())
        self.wq = wq   # qint8 packed
        self._bias = bias.float() if bias is not None else None
    def forward(self, x):
        w = self.wq.dequantize().to(torch.bfloat16)
        y = torch.nn.functional.linear(x.to(torch.bfloat16) * self.sinv.to(torch.bfloat16), w)
        if self._bias is not None:
            y = y + self._bias.to(torch.bfloat16)
        return y

class SQLinear(qdyn.Linear):
    """SmoothQuant W8A8: x/s → int8 动态激活(reduce_range=False) × int8 per-channel 权重"""
    def __init__(self, in_f, out_f, wq, s_in, bias):
        super().__init__(in_f, out_f, bias_=bias is not None)
        self.set_weight_bias(wq, bias.float() if bias is not None else None)
        self.register_buffer("sinv", (1.0 / s_in).float())
    def forward(self, x):
        xf = x.float() * self.sinv
        y = torch.ops.quantized.linear_dynamic(
            xf, self._packed_params._packed_params, reduce_range=False)
        return y.to(torch.bfloat16)

def build_sq(alpha):
    cfg = json.load(open(f"{ENC_DIR}/config.json")); cfg.pop("quantization_config", None)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg))
    torch.set_default_dtype(old)
    with safe_open(f"{ENC_DIR}/model.safetensors", framework="pt") as f:
        for n, m in list(enc.named_modules()):
            if not isinstance(m, nn.Linear) or (n + ".weight") not in cal and True:
                if not isinstance(m, nn.Linear):
                    continue
            if n not in cal:
                continue   # visual/其他, 不量化
            ratio = (cal[n].max() / cal[n].median().clamp(min=1e-6)).item()
            use_a16 = MIXED and ratio > RATIO_TH
            k = n + ".weight"
            W = f.get_tensor(k)
            if W.dtype == torch.float8_e4m3fn:
                W = (W.to(torch.float32) * f.get_tensor(k + "_scale")).to(torch.bfloat16)
            bias = None
            try:
                bias = f.get_tensor(n + ".bias")
            except Exception:
                pass
            # smooth: s_j = maxX^α / maxW^(1-α); W' = W*s; x' = x/s
            Wf = W.float()
            maxW = Wf.abs().amax(dim=0).clamp(min=1e-8)      # [K] 列 absmax
            maxX = cal[n].clamp(min=1e-8)                    # [K]
            s = (maxX ** alpha) / (maxW ** (1 - alpha))
            W2 = Wf * s.unsqueeze(0)
            scale_row = W2.abs().amax(dim=1) / 127.0
            q = torch.round(W2 / scale_row.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
            z = torch.zeros_like(scale_row, dtype=torch.int32)
            wq = torch._make_per_channel_quantized_tensor(q, scale_row, z, 0)
            parts = n.split("."); parent = enc
            for x_ in parts[:-1]:
                parent = getattr(parent, x_)
            if use_a16:
                setattr(parent, parts[-1], W16Linear(wq, s, bias))
            else:
                setattr(parent, parts[-1], SQLinear(m.in_features, m.out_features, wq, s, bias))
        # 非 Linear 层 bf16 装载
        keys = set(f.keys())
        for name, module in enc.named_modules():
            for pname in [pn for pn, _ in module.named_parameters(recurse=False)]:
                full = f"{name}.{pname}" if name else pname
                if full + "_scale" in keys or full.startswith("model.visual.") or full.startswith("lm_head."):
                    continue
                try:
                    t = f.get_tensor(full)
                except Exception:
                    continue
                if t.dtype == torch.float8_e4m3fn:
                    continue
                setattr(module, pname, nn.Parameter(t.to(torch.bfloat16), requires_grad=False))
    for m in enc.modules():
        inv = getattr(m, "inv_freq", None)
        if inv is not None and inv.is_meta:
            dim = inv.shape[-1] * 2
            m.inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return enc.eval()

for alpha in ALPHAS:
    enc = build_sq(alpha)
    n8 = sum(1 for m in enc.modules() if isinstance(m, SQLinear))
    n16 = sum(1 for m in enc.modules() if isinstance(m, W16Linear))
    print(f"构成: W8A8 {n8} 层 / W8A16 {n16} 层")
    import psutil
    rss = psutil.Process().memory_info().rss / 2**30
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(TPL.format(p), return_tensors="pt")["input_ids"]
            t0 = time.time()
            out = enc(ids, output_hidden_states=True, use_cache=False)
            dt = time.time() - t0
            h8 = out.hidden_states[-1][0].float().cpu()
            hb = base[p]
            cos = torch.nn.functional.cosine_similarity(h8.flatten().unsqueeze(0),
                                                        hb.flatten().unsqueeze(0)).item()
            rel = ((h8 - hb).norm() / hb.norm()).item()
            print(f"α={alpha:.2f} [{p[:18]}...]: {dt:.2f}s cos={cos:.6f} relerr={rel:.4f}")
    del enc, out
    gc.collect()
print("SQ_VERIFY_DONE")
