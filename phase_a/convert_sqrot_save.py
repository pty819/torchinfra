# 全模型两级旋转 W8A8 验证 (无 smooth, 纯旋转 + per-token 动态量化)
import json, time, sys, gc, torch
sys.path.insert(0, "~/项目/qwen21dit/phase_a")
import torch.nn as nn
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from safetensors.torch import safe_open
from encoder import load_encoder, TPL

ENC_DIR = "~/models/qwen21/fp8/qwen3vl8b_fp8"
PROMPTS = ["masterpiece, a serene mountain lake at dawn, mist over water, ultra detailed",
           "1girl, cute anime style, cherry blossoms, golden hour lighting, ultra detailed",
           " masterpiece, best quality, 1girl, seductive pose, dim lighting, detailed skin"]
N0 = 256
SEED = 20261122

torch.manual_seed(SEED)

def reg_hadamard_4():
    return torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                         [1., -1., 1., 1.], [-1., 1., 1., 1.]])

def kron(a, b):
    return torch.einsum("ij,kl->ikjl", a, b).reshape(a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])

def reg4():
    return torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                         [1., -1., 1., 1.], [-1., 1., 1., 1.]])

def kron2(a, b):
    return torch.einsum("ij,kl->ikjl", a, b).reshape(a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])

# R256: 与 learn_rot 完全一致 (全局 seed 第一次 randn 的符号)
torch.manual_seed(SEED)
_h = reg4()
for _ in range(3):
    _h = kron2(_h, reg4())
R256 = torch.diag(torch.randn(256).sign()) @ (_h / 16.0)

from safetensors.torch import safe_open as _so
_HQ = {}
with _so("~/models/qwen21/int8_qwen3vl8b.rotlearn.safetensors", framework="pt") as _f:
    for _k in _f.keys():
        _HQ[_k[:-3]] = _f.get_tensor(_k).float()

import os as _os
ALPHA = float(_os.environ.get("SQ_ALPHA", "0.5"))
_CAL = {}
from safetensors.torch import safe_open as _so2
with _so2("~/models/qwen21/int8_qwen3vl8b.calib.safetensors", framework="pt") as _f:
    for _k in _f.keys():
        _CAL[_k] = _f.get_tensor(_k)

def build_rot(name, K):
    g = torch.Generator().manual_seed(SEED + K)
    NG = K // N0
    Hq = torch.linalg.qr(torch.randn(NG, NG, generator=g))[0]
    Hq = Hq * torch.randn(NG, generator=g).sign()
    return Hq.float(), R256

def two_level_rot(x, Hq=None):
    assert Hq is not None
    K = x.shape[-1]
    NG = K // N0
    shp = x.shape
    xf = x.reshape(-1, K).float()
    b = xf.shape[0]
    x1 = torch.einsum("bgn,hg->bhn", xf.view(b, NG, N0), Hq)
    return (x1.reshape(b, N0) @ R256).reshape(shp) if False else (x1.reshape(-1, N0) @ R256).reshape(shp)

class RLinear(torch.nn.Module):
    """两级旋转 W8A8: x 旋转 -> int8 动态量化(reduce_range=False) x int8 per-channel 旋转权重"""
    def __init__(self, in_f, out_f, wq, bias, Hq, sinv):
        super().__init__()
        import torch.ao.nn.quantized.dynamic as qdyn
        self._core = qdyn.Linear(in_f, out_f, bias_=bias is not None)
        self._core.set_weight_bias(wq, bias.float() if bias is not None else None)
        self.register_buffer("Hq", Hq)
        self.register_buffer("sinv", sinv)
    def forward(self, x):
        xr = two_level_rot(x.float() * self.sinv, self.Hq)
        y = torch.ops.quantized.linear_dynamic(
            xr, self._core._packed_params._packed_params, reduce_range=False)
        return y.to(torch.bfloat16)

tok = AutoTokenizer.from_pretrained(ENC_DIR)
tok_, enc_bf, _, _ = load_encoder()
base = {}
with torch.no_grad():
    for p in PROMPTS:
        ids = tok_(TPL.format(p), return_tensors="pt")["input_ids"]
        out = enc_bf(ids, output_hidden_states=True, use_cache=False)
        base[p] = out.hidden_states[-1][0].float().cpu()
del tok_, enc_bf, out
gc.collect()

cfg = json.load(open(f"{ENC_DIR}/config.json")); cfg.pop("quantization_config", None)
old = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
with torch.device("meta"):
    enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg))
torch.set_default_dtype(old)

import time as _t
OUTQ = {}
t0 = _t.time()
nq = 0
with safe_open(f"{ENC_DIR}/model.safetensors", framework="pt") as f:
    for n, m in list(enc.named_modules()):
        if not isinstance(m, nn.Linear):
            continue
        if n.startswith("model.language_model") is False:
            continue
        import os as _os
        if _os.environ.get("SKIP_L17") == "1" and ".layers.17." in n:
            continue
        W = f.get_tensor(n + ".weight")
        if W.dtype == torch.float8_e4m3fn:
            W = (W.to(torch.float32) * f.get_tensor(n + ".weight_scale")).to(torch.bfloat16)
        bias = None
        try:
            bias = f.get_tensor(n + ".bias")
        except Exception:
            pass
        Hq_l = build_rot(n, W.shape[1])[0]
        maxX = _CAL[n].clamp(min=1e-8)
        maxW = W.float().abs().amax(dim=0).clamp(min=1e-8)
        ssm = (maxX ** ALPHA) / (maxW ** (1 - ALPHA))
        sinv = (1.0 / ssm).float()
        Wr = two_level_rot(W.float() * ssm, Hq_l)   # smooth 迁移 + 旋转
        s = Wr.abs().amax(dim=1) / 127.0
        q = torch.round(Wr / s.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        z = torch.zeros_like(s, dtype=torch.int32)
        wq = torch._make_per_channel_quantized_tensor(q, s, z, 0)
        OUTQ.setdefault(n, {})
        OUTQ[n]["int8"] = q.cpu().contiguous()
        OUTQ[n]["scale"] = s.cpu().contiguous()
        OUTQ[n]["sinv"] = sinv.cpu().contiguous()
        if bias is not None:
            OUTQ[n]["bias"] = bias.to(torch.bfloat16).cpu().contiguous()
        parts = n.split("."); parent = enc
        for x_ in parts[:-1]:
            parent = getattr(parent, x_)
        setattr(parent, parts[-1], RLinear(m.in_features, m.out_features, wq, bias, Hq_l, sinv))
        nq += 1
        del W, Wr, q
    keys = set(f.keys())
    for name, module in enc.named_modules():
        for pname in [pn for pn, _ in module.named_parameters(recurse=False)]:
            full = f"{name}.{pname}" if name else pname
            if full in keys and not full.startswith("model.visual.") and not full.startswith("lm_head."):
                t = f.get_tensor(full)
                if t.dtype == torch.float8_e4m3fn:
                    if (full + "_scale") not in keys:
                        continue
                    t = (t.to(torch.float32) * f.get_tensor(full + "_scale")).to(torch.bfloat16)
                setattr(module, pname, nn.Parameter(t.to(torch.bfloat16), requires_grad=False))
print(f"替换 {nq} 层, 转换 {_t.time()-t0:.0f}s")
for m in enc.modules():
    inv = getattr(m, "inv_freq", None)
    if inv is not None and inv.is_meta:
        dim = inv.shape[-1] * 2
        m.inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
enc = enc.eval()

import psutil
print(f"RSS: {psutil.Process().memory_info().rss/2**30:.1f}GB")
with torch.no_grad():
    # 基线也要逐层
    tokb, encb, _, _ = load_encoder()
    p = PROMPTS[0]
    ids = tok(TPL.format(p), return_tensors="pt")["input_ids"]
    outb = encb(ids, output_hidden_states=True, use_cache=False)
    out = enc(ids, output_hidden_states=True, use_cache=False)
    for i, (ha, hb2) in enumerate(zip(out.hidden_states, outb.hidden_states)):
        c = torch.nn.functional.cosine_similarity(
            ha[0].float().flatten().unsqueeze(0), hb2[0].float().flatten().unsqueeze(0)).item()
        if i == len(out.hidden_states) - 1:
            print(f"ALPHA={ALPHA} 终点 cos={c:.6f}")
from safetensors.torch import save_file as _sf
out = {}
for n, d in OUTQ.items():
    for k, v in d.items():
        out[n + "." + k] = v
_sf(out, "~/models/qwen21/int8_qwen3vl8b_sqrot.safetensors")
print("SAVED", len(out))
print("RHT_DIAG_DONE")
