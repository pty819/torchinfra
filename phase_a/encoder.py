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


SQ_FILE = "~/models/qwen21/int8_qwen3vl8b_sq.safetensors"

def load_encoder_sq():
    """W8A8 SmoothQuant(alpha=0.75 离线校准) encoder: int8 权重常驻 + VNNI int8 算力。
    权重已离线迁移离群(s 融入权重行), forward 激活 *sinv 后 int8 动态量化(reduce_range=False)。
    返回 (tok, enc, lm, drop), 接口与 load_encoder 一致。RSS ~9.9GB vs bf16 全量 ~17GB。"""
    import torch.nn as nn
    from safetensors.torch import safe_open
    import torch.ao.nn.quantized.dynamic as qdyn

    tok = AutoTokenizer.from_pretrained(ENC_DIR)
    cfg = json.load(open(f"{ENC_DIR}/config.json")); cfg.pop("quantization_config", None)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg))
    torch.set_default_dtype(old)

    sd = {}
    with safe_open(SQ_FILE, framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
        vis_keys = [k for k in f.keys() if ".visual." in k]
    for k in vis_keys:   # visual 塔不装, 留 meta 省内存
        sd.pop(k, None)

    class SQLinear(qdyn.Linear):
        def __init__(self, in_f, out_f, wq, sinv, bias):
            super().__init__(in_f, out_f, bias_=bias is not None)
            self.set_weight_bias(wq, bias.float() if bias is not None else None)
            self.register_buffer("sinv", sinv)
        def forward(self, x):
            xf = x.float() * self.sinv
            y = torch.ops.quantized.linear_dynamic(
                xf, self._packed_params._packed_params, reduce_range=False)
            return y.to(torch.bfloat16)

    def parent_set(root, name, val):
        parts = name.split(".")
        p = root
        for x_ in parts[:-1]:
            p = getattr(p, x_)
        setattr(p, parts[-1], val)

    for n, m in list(enc.named_modules()):
        if not isinstance(m, torch.nn.Linear):
            continue
        k = n + ".weight"
        if (k + ".int8") not in sd:
            continue
        z = torch.zeros_like(sd[k + ".scale"], dtype=torch.int32)
        wq = torch._make_per_channel_quantized_tensor(
            sd[k + ".int8"], sd[k + ".scale"], z, 0)
        parent_set(enc, n, SQLinear(m.in_features, m.out_features, wq,
                                    sd[k + ".sinv"], sd.get(n + ".bias.bf16")))
    for name, module in enc.named_modules():
        for pname in [pn for pn, _ in module.named_parameters(recurse=False)]:
            full = f"{name}.{pname}" if name else pname
            if (full + ".bf16") in sd and (full + ".weight.int8") not in sd:
                setattr(module, pname, nn.Parameter(sd[full + ".bf16"], requires_grad=False))
    for m in enc.modules():
        inv = getattr(m, "inv_freq", None)
        if inv is not None and inv.is_meta:
            dim = inv.shape[-1] * 2
            m.inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    enc = enc.eval()
    drop = tok(SYS_TXT, return_tensors="pt")["input_ids"].shape[1]
    return tok, enc, enc.model.language_model, drop


SQROT = "~/models/qwen21/int8_qwen3vl8b_sqrot.safetensors"
_ROT_SEED = 20261122
_N0 = 256

def _r256():
    import torch as _t
    _t.manual_seed(_ROT_SEED)
    h = _t.tensor([[1.,1.,1.,-1.],[1.,1.,-1.,1.],[1.,-1.,1.,1.],[-1.,1.,1.,1.]])
    for _ in range(3):
        a, b = h.shape; h = _t.einsum("ij,kl->ikjl", h, _t.tensor([[1.,1.,1.,-1.],[1.,1.,-1.,1.],[1.,-1.,1.,1.],[-1.,1.,1.,1.]])).reshape(a*4, b*4)
    return _t.diag(_t.randn(256).sign()) @ (h / 16.0), h

def load_encoder_sqrot():
    """纯 W8A8 组合版: SmoothQuant(a=0.5 离线迁移) x 两级随机旋转, cos 0.9979"""
    import torch.nn as nn
    from safetensors.torch import safe_open
    import torch.ao.nn.quantized.dynamic as qdyn
    tok = AutoTokenizer.from_pretrained(ENC_DIR)
    cfg = json.load(open(f"{ENC_DIR}/config.json")); cfg.pop("quantization_config", None)
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg))
    torch.set_default_dtype(old)
    sd = {}
    with safe_open(SQROT, framework="pt") as f:
        for k in f.keys():
            if ".visual." not in k:
                sd[k] = f.get_tensor(k)
    with safe_open(SQ_FILE, framework="pt") as f:   # bf16 杂层从 sq 包补
        for k in f.keys():
            if k.endswith(".bf16") and k not in sd and ".visual." not in k:
                sd[k] = f.get_tensor(k)
    _R = _r256()[0]
    _HQ = {}
    def hq_for(K):
        if K not in _HQ:
            g = torch.Generator().manual_seed(_ROT_SEED + K)
            NG = K // _N0
            H = torch.linalg.qr(torch.randn(NG, NG, generator=g))[0]
            _HQ[K] = (H * torch.randn(NG, generator=g).sign()).float()
        return _HQ[K]
    def rot2(x, Hq):
        K = x.shape[-1]; NG = K // _N0; shp = x.shape
        xf = x.reshape(-1, K).float(); b = xf.shape[0]
        x1 = torch.matmul(xf.view(b, NG, _N0).transpose(1, 2), Hq.t()).transpose(1, 2)
        return (x1.reshape(-1, _N0) @ _R).reshape(shp)
    class RL(qdyn.Linear):
        def __init__(self, i_, o_, wq, sinv, bias):
            super().__init__(i_, o_, bias_=bias is not None)
            self.set_weight_bias(wq, bias.float() if bias is not None else None)
            self.register_buffer("sinv", sinv)
        def forward(self, x):
            xr = rot2(x.float() * self.sinv, hq_for(x.shape[-1]))
            return torch.ops.quantized.linear_dynamic(
                xr, self._packed_params._packed_params, reduce_range=False).to(torch.bfloat16)
    def pset(root, name, val):
        p = root
        for x_ in name.split(".")[:-1]:
            p = getattr(p, x_)
        setattr(p, name.split(".")[-1], val)
    for n, m in list(enc.named_modules()):
        if not isinstance(m, torch.nn.Linear) or (n + ".int8") not in sd:
            continue
        z = torch.zeros_like(sd[n + ".scale"], dtype=torch.int32)
        wq = torch._make_per_channel_quantized_tensor(sd[n + ".int8"], sd[n + ".scale"], z, 0)
        pset(enc, n, RL(m.in_features, m.out_features, wq, sd[n + ".sinv"], sd.get(n + ".bias")))
    for name, module in enc.named_modules():
        for pname in [pn for pn, _ in module.named_parameters(recurse=False)]:
            full = f"{name}.{pname}" if name else pname
            if (full + ".bf16") in sd and not any(full.startswith(q) for q in ("model.visual.", "lm_head.")):
                setattr(module, pname, nn.Parameter(sd[full + ".bf16"], requires_grad=False))
    for m in enc.modules():
        inv = getattr(m, "inv_freq", None)
        if inv is not None and inv.is_meta:
            dim = inv.shape[-1] * 2
            m.inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    enc = enc.eval()
    drop = tok(SYS_TXT, return_tensors="pt")["input_ids"].shape[1]
    return tok, enc, enc.model.language_model, drop
