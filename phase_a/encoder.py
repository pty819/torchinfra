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
    _GPU = {"R": None, "HQ": {}}
    class RL(qdyn.Linear):
        """旋转在 GPU 执行 (encode 时 GPU 空闲): 激活 38x4096 微小, 传输+旋转 <0.3ms/层,
        CPU 上同形状小 matmul 被 16 线程 fork/join 拖到 ~10ms/层"""
        def __init__(self, i_, o_, wq, sinv, bias):
            super().__init__(i_, o_, bias_=bias is not None)
            self.set_weight_bias(wq, bias.float() if bias is not None else None)
            self.register_buffer("sinv", sinv)
        def forward(self, x):
            K = x.shape[-1]; NG = K // _N0
            if _GPU["R"] is None:
                _GPU["R"] = _R.float().cuda()
            if K not in _GPU["HQ"]:
                _GPU["HQ"][K] = hq_for(K).cuda()
                _GPU["HQ"][("sinv", K)] = None
            xg = x.float().cuda().reshape(-1, K)
            b = xg.shape[0]
            a = (xg * self.sinv.cuda()).view(b, NG, _N0).transpose(1, 2) @ _GPU["HQ"][K].t()
            xr = (a.transpose(1, 2).reshape(-1, _N0) @ _GPU["R"]).reshape(b, K).cpu()
            y = torch.ops.quantized.linear_dynamic(
                xr, self._packed_params._packed_params, reduce_range=False).to(torch.bfloat16)
            return y.reshape(*x.shape[:-1], -1)
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


# ---------------- sqrot2: 全蝶形 CPU 旋转 (方案1) + qkv/gateup 组入口合并 (方案2) ----------------
# 组内 B4^⊗4 蝶形 + 组间 B4^⊗2 / O3⊗B4^⊗2 蝶形, 无小 matmul; 旋转 252 -> 144 次, 纯 CPU 无 PCIe
SQROT2 = "~/models/qwen21/int8_qwen3vl8b_sqrot2.safetensors"
_B4 = torch.tensor([[1.,1.,1.,-1.],[1.,1.,-1.,1.],[1.,-1.,1.,1.],[-1.,1.,1.,1.]])
_O3r = torch.tensor([[1.,1.,1.],[2.,-1.,-1.],[0.,1.,-1.]])
_O3 = (_O3r / _O3r.norm(dim=1, keepdim=True)).float()
_C0, _C1, _C2 = 0.5773502691896258, 0.4082482904638631, 0.7071067811865476

def _kron(a, b):
    return torch.einsum("ij,kl->ikjl", a, b).reshape(a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])

def _bf4(x):
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return torch.stack([x0+x1+x2-x3, x0+x1-x2+x3, x0-x1+x2+x3, -x0+x1+x2+x3], -1)

def _bf3(x):
    x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    return torch.stack([(x0+x1+x2)*_C0, (2*x0-x1-x2)*_C1, (x1-x2)*_C2], -1)

def _bf_intra(x):                                  # [N,256] -> x @ (B4^⊗4/16)
    T = x.shape[0]
    v = x.view(T, 4, 4, 4, 4)
    v = _bf4(v.permute(0, 2, 3, 4, 1).contiguous().view(-1, 4)).view(T, 4, 4, 4, 4).permute(0, 4, 1, 2, 3)
    v = _bf4(v.permute(0, 1, 3, 4, 2).contiguous().view(-1, 4)).view(T, 4, 4, 4, 4).permute(0, 1, 4, 2, 3)
    v = _bf4(v.permute(0, 1, 2, 4, 3).contiguous().view(-1, 4)).view(T, 4, 4, 4, 4).permute(0, 1, 2, 4, 3)
    v = _bf4(v.reshape(-1, 4)).view(T, 4, 4, 4, 4)
    return v.reshape(T, 256) / 16.0

def _inter_bf(t, NG):                              # t [b,NG,256], NG 轴蝶形
    b = t.shape[0]
    if NG == 16:
        v = t.permute(0, 2, 1).contiguous().view(b, 256, 4, 4)
        v = _bf4(v.reshape(-1, 4)).view(b, 256, 4, 4)
        v = _bf4(v.permute(0, 1, 3, 2).contiguous().view(-1, 4)).view(b, 256, 4, 4).permute(0, 1, 3, 2)
        return v.reshape(b, 256, 16).permute(0, 2, 1).contiguous() / 4.0
    v = t.permute(0, 2, 1).contiguous().view(b, 256, 3, 4, 4)   # NG=48
    v = _bf4(v.reshape(-1, 4)).view(b, 256, 3, 4, 4)
    v = _bf4(v.permute(0, 1, 2, 4, 3).contiguous().view(-1, 4)).view(b, 256, 3, 4, 4).permute(0, 1, 2, 4, 3)
    v = _bf3(v.permute(0, 1, 3, 4, 2).contiguous().view(-1, 3)).view(b, 256, 4, 4, 3).permute(0, 1, 4, 2, 3)
    return v.reshape(b, 256, 48).permute(0, 2, 1).contiguous() / 4.0

_INTER2 = {}
def _inter2(K):
    """组间随机符号 (d1 输出侧, d2 输入侧折进 sinv); 矩阵 D1·Hb·D2 不显式构造"""
    if K not in _INTER2:
        g = torch.Generator().manual_seed(_ROT_SEED + K + 7)
        d1 = torch.randn(K // _N0, generator=g).sign()
        d2 = torch.randn(K // _N0, generator=g).sign()
        _INTER2[K] = (d1, d2.repeat_interleave(_N0))
    return _INTER2[K]

_gI = torch.Generator().manual_seed(_ROT_SEED)
_SINTRA = torch.randn(256, generator=_gI).sign()   # x@(diag(s)@H/16) = (x*s)@H/16 (H 对称)

def _rot_online(x2d, sinv, d1, d2c):
    """[b,K] fp32 -> 旋转后 [b,K] fp32; 纯加减蝶形, 无 matmul"""
    NG = x2d.shape[1] // _N0
    t = x2d.view(-1, NG, _N0) * (sinv * d2c).view(1, NG, _N0)
    t = _inter_bf(t, NG) * d1.view(1, NG, 1)
    return _bf_intra(t.reshape(-1, _N0) * _SINTRA).reshape(x2d.shape[0], x2d.shape[1])

_ROTC = torch.compile(_rot_online, dynamic=True)

def load_encoder_sqrot2():
    """sqrot + 全蝶形 CPU 旋转 + 共享组入口合并; 权重 int8_qwen3vl8b_sqrot2.safetensors"""
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
    with safe_open(SQROT2, framework="pt") as f:
        for k in f.keys():
            if ".visual." not in k:
                sd[k] = f.get_tensor(k)
    with safe_open(SQ_FILE, framework="pt") as f:   # bf16 杂层仍从 sq 包补
        for k in f.keys():
            if k.endswith(".bf16") and k not in sd and ".visual." not in k:
                sd[k] = f.get_tensor(k)

    class RL2S(qdyn.Linear):
        """共享组成员 (q/k/v, gate/up): 输入已被 RotNorm 预旋转, 直接 int8 GEMM"""
        def __init__(self, i_, o_, wq, bias):
            super().__init__(i_, o_, bias_=bias is not None)
            self.set_weight_bias(wq, bias.float() if bias is not None else None)
        def forward(self, x):
            y = torch.ops.quantized.linear_dynamic(
                x.reshape(-1, x.shape[-1]), self._packed_params._packed_params, reduce_range=False)
            return y.to(torch.bfloat16).reshape(*x.shape[:-1], -1)

    class RL2(qdyn.Linear):
        """独立输入 (o_proj/down_proj): 自带蝶形旋转"""
        def __init__(self, i_, o_, wq, bias, sinv):
            super().__init__(i_, o_, bias_=bias is not None)
            self.set_weight_bias(wq, bias.float() if bias is not None else None)
            self.register_buffer("sinv", sinv)
        def forward(self, x):
            K = x.shape[-1]
            d1, d2c = _inter2(K)
            xr = _ROTC(x.reshape(-1, K).float(), self.sinv, d1, d2c)
            y = torch.ops.quantized.linear_dynamic(
                xr, self._packed_params._packed_params, reduce_range=False)
            return y.to(torch.bfloat16).reshape(*x.shape[:-1], -1)

    class RotNorm(torch.nn.Module):
        """RMSNorm 出口接一次共享旋转: qkv 组 3->1, gateup 组 2->1"""
        def __init__(self, norm, sinv):
            super().__init__()
            self.norm, self.sinv = norm, sinv
        def forward(self, x):
            y = self.norm(x)
            K = y.shape[-1]
            d1, d2c = _inter2(K)
            return _ROTC(y.reshape(-1, K).float(), self.sinv, d1, d2c).reshape(*y.shape)

    def pset(root, name, val):
        p = root
        for x_ in name.split(".")[:-1]:
            p = getattr(p, x_)
        setattr(p, name.split(".")[-1], val)

    SHARED = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    for n, m in list(enc.named_modules()):
        if not isinstance(m, torch.nn.Linear) or (n + ".int8") not in sd:
            continue
        z = torch.zeros_like(sd[n + ".scale"], dtype=torch.int32)
        wq = torch._make_per_channel_quantized_tensor(sd[n + ".int8"], sd[n + ".scale"], z, 0)
        if n.split(".")[-1] in SHARED:
            pset(enc, n, RL2S(m.in_features, m.out_features, wq, sd.get(n + ".bias")))
        else:
            pset(enc, n, RL2(m.in_features, m.out_features, wq, sd.get(n + ".bias"), sd[n + ".sinv"]))
    for name, module in enc.named_modules():
        for pname in [pn for pn, _ in module.named_parameters(recurse=False)]:
            full = f"{name}.{pname}" if name else pname
            if (full + ".bf16") in sd and not any(full.startswith(q) for q in ("model.visual.", "lm_head.")):
                setattr(module, pname, nn.Parameter(sd[full + ".bf16"], requires_grad=False))
    for n, m in list(enc.named_modules()):          # 组入口: norm -> norm+旋转
        last = n.split(".")[-1]
        if last == "input_layernorm" and ".layers." in n:
            pre = n.rsplit(".", 1)[0]
            pset(enc, n, RotNorm(m, sd[pre + ".self_attn.q_proj.sinv"]))
        elif last == "post_attention_layernorm" and ".layers." in n:
            pre = n.rsplit(".", 1)[0]
            pset(enc, n, RotNorm(m, sd[pre + ".mlp.gate_proj.sinv"]))
    for m in enc.modules():
        inv = getattr(m, "inv_freq", None)
        if inv is not None and inv.is_meta:
            dim = inv.shape[-1] * 2
            m.inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    enc = enc.eval()
    with torch.no_grad():                           # 编译预热 (dynamic, 两个 K 变体)
        _ROTC(torch.randn(4, 4096), torch.ones(4096), *_inter2(4096))
        _ROTC(torch.randn(4, 12288), torch.ones(12288), *_inter2(12288))
    drop = tok(SYS_TXT, return_tensors="pt")["input_ids"].shape[1]
    return tok, enc, enc.model.language_model, drop
