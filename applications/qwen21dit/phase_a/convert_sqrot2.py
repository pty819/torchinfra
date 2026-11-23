# sqrot2: 组间蝶形矩阵 (B4^⊗2 / O3⊗B4^⊗2) + qkv/gateup 组共享 smooth s, 在线全蝶形 W8A8
# cos 达标 (>=0.9975) 才落盘 int8_qwen3vl8b_sqrot2.safetensors
from paths import MODEL_DIR
import json, time, sys, gc, os, torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from safetensors.torch import safe_open, save_file
from encoder import load_encoder, TPL

ENC_DIR = str(MODEL_DIR / "fp8/qwen3vl8b_fp8")
OUT = str(MODEL_DIR / "int8_qwen3vl8b_sqrot2.safetensors")
PROMPTS = ["masterpiece, a serene mountain lake at dawn, mist over water, ultra detailed",
           "1girl, cute anime style, cherry blossoms, golden hour lighting, ultra detailed",
           " masterpiece, best quality, 1girl, seductive pose, dim lighting, detailed skin"]
N0, SEED = 256, 20261122
ALPHA = float(os.environ.get("SQ_ALPHA", "0.5"))
COS_GATE = float(os.environ.get("COS_GATE", "0.9975"))

B4 = torch.tensor([[1.,1.,1.,-1.],[1.,1.,-1.,1.],[1.,-1.,1.,1.],[-1.,1.,1.,1.]])
O3r = torch.tensor([[1.,1.,1.],[2.,-1.,-1.],[0.,1.,-1.]])
O3 = (O3r / O3r.norm(dim=1, keepdim=True)).float()
C0, C1, C2 = 0.5773502691896258, 0.4082482904638631, 0.7071067811865476

def kron(a, b):
    return torch.einsum("ij,kl->ikjl", a, b).reshape(a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])

def bf4(x):
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return torch.stack([x0+x1+x2-x3, x0+x1-x2+x3, x0-x1+x2+x3, -x0+x1+x2+x3], -1)

def bf3(x):
    x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    return torch.stack([(x0+x1+x2)*C0, (2*x0-x1-x2)*C1, (x1-x2)*C2], -1)

def bf_intra(x):                           # [N,256] -> x @ (B4^⊗4/16)
    T = x.shape[0]
    v = x.view(T, 4, 4, 4, 4)
    v = bf4(v.permute(0, 2, 3, 4, 1).contiguous().view(-1, 4)).view(T, 4, 4, 4, 4).permute(0, 4, 1, 2, 3)
    v = bf4(v.permute(0, 1, 3, 4, 2).contiguous().view(-1, 4)).view(T, 4, 4, 4, 4).permute(0, 1, 4, 2, 3)
    v = bf4(v.permute(0, 1, 2, 4, 3).contiguous().view(-1, 4)).view(T, 4, 4, 4, 4).permute(0, 1, 2, 4, 3)
    v = bf4(v.reshape(-1, 4)).view(T, 4, 4, 4, 4)
    return v.reshape(T, 256) / 16.0

def inter_bf(t, NG):                       # t [b,NG,256] -> Hb 作用于 NG 轴
    b = t.shape[0]
    if NG == 16:
        v = t.permute(0, 2, 1).contiguous().view(b, 256, 4, 4)
        v = bf4(v.reshape(-1, 4)).view(b, 256, 4, 4)
        v = bf4(v.permute(0, 1, 3, 2).contiguous().view(-1, 4)).view(b, 256, 4, 4).permute(0, 1, 3, 2)
        return v.reshape(b, 256, 16).permute(0, 2, 1).contiguous() / 4.0
    elif NG == 48:
        v = t.permute(0, 2, 1).contiguous().view(b, 256, 3, 4, 4)
        v = bf4(v.reshape(-1, 4)).view(b, 256, 3, 4, 4)
        v = bf4(v.permute(0, 1, 2, 4, 3).contiguous().view(-1, 4)).view(b, 256, 3, 4, 4).permute(0, 1, 2, 4, 3)
        v = bf3(v.permute(0, 1, 3, 4, 2).contiguous().view(-1, 3)).view(b, 256, 4, 4, 3).permute(0, 1, 4, 2, 3)
        return v.reshape(b, 256, 48).permute(0, 2, 1).contiguous() / 4.0
    raise ValueError(NG)

_INTER = {}
def inter_mat(K):
    if K not in _INTER:
        NG = K // N0
        g = torch.Generator().manual_seed(SEED + K + 7)
        d1 = torch.randn(NG, generator=g).sign()
        d2 = torch.randn(NG, generator=g).sign()
        Hb = kron(B4, B4) / 4.0 if NG == 16 else kron(O3, kron(B4, B4)) / 4.0
        _INTER[K] = (torch.diag(d1) @ Hb @ torch.diag(d2), d1, d2)
    return _INTER[K]

torch.manual_seed(SEED)
_h = B4.clone()
for _ in range(3):
    _h = kron(_h, B4)
S_INTRA = torch.randn(256).sign()          # x @ (diag(s)@H) = (x*s) @ H
R_INTRA = torch.diag(S_INTRA) @ (_h / 16.0)

def rot_bf(x, sinv, K):                    # 在线: 全蝶形 (sinv 原始, d2 折叠进乘)
    NG = K // N0
    b = x.reshape(-1, K).shape[0]
    _, d1, d2 = inter_mat(K)
    t = x.reshape(b, NG, N0).float() * (sinv * d2.repeat_interleave(N0)).view(1, NG, N0)
    t = inter_bf(t, NG) * d1.view(1, NG, 1)
    return bf_intra(t.reshape(-1, N0) * S_INTRA).reshape(b, K)

def rot_w(W, ssm, K):                      # 离线权重: matmul 形式 (与在线蝶形等价, bench 已证 3.6e-6)
    NG = K // N0
    M = inter_mat(K)[0]
    a = torch.einsum("bgn,hg->bhn", (W * ssm).view(-1, NG, N0), M)
    return (a.reshape(-1, N0) @ R_INTRA).reshape(W.shape)

_CAL = {}
with safe_open(str(MODEL_DIR / "int8_qwen3vl8b.calib.safetensors"), framework="pt") as f:
    for k in f.keys():
        _CAL[k] = f.get_tensor(k)

def group_of(n):
    parts = n.split(".")
    if parts[-1] in ("q_proj", "k_proj", "v_proj"):
        return ".".join(parts[:-1]) + ".qkv"
    if parts[-1] in ("gate_proj", "up_proj"):
        return ".".join(parts[:-1]) + ".gateup"
    return n

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
old = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
with torch.device("meta"):
    enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg))
torch.set_default_dtype(old)

import torch.ao.nn.quantized.dynamic as qdyn
class RLinear(torch.nn.Module):
    def __init__(self, i_, o_, wq, bias, sinv):
        super().__init__()
        self._core = qdyn.Linear(i_, o_, bias_=bias is not None)
        self._core.set_weight_bias(wq, bias.float() if bias is not None else None)
        self.register_buffer("sinv", sinv)
    def forward(self, x):
        K = x.shape[-1]
        xr = rot_bf(x, self.sinv, K)
        y = torch.ops.quantized.linear_dynamic(
            xr, self._core._packed_params._packed_params, reduce_range=False)
        return y.to(torch.bfloat16).reshape(*x.shape[:-1], -1)

# ---- pass1: 各 linear maxW ----
t0 = time.time()
maxW = {}
with safe_open(f"{ENC_DIR}/model.safetensors", framework="pt") as f:
    for n, m in enc.named_modules():
        if isinstance(m, nn.Linear) and n.startswith("model.language_model"):
            W = f.get_tensor(n + ".weight")
            if W.dtype == torch.float8_e4m3fn:
                W = W.to(torch.float32) * f.get_tensor(n + ".weight_scale")
            maxW[n] = W.float().abs().amax(dim=0).clamp(min=1e-8)
            del W
print(f"pass1 maxW {time.time()-t0:.0f}s")

# ---- 组共享 ssm ----
groups = {}
for n in maxW:
    groups.setdefault(group_of(n), []).append(n)
SSM = {}
for gname, members in groups.items():
    if len(members) == 1:
        mx = _CAL[members[0]].clamp(min=1e-8)
        gm = maxW[members[0]]
    else:
        mx = _CAL[members[0]].clamp(min=1e-8)          # q/k/v (gate/up) 输入相同
        gm = torch.stack([maxW[m] for m in members]).prod(dim=0) ** (1.0 / len(members))
    SSM[gname] = (mx ** ALPHA) / (gm ** (1 - ALPHA))
print(f"组数 {len(SSM)} (共享组: {sum(1 for v in groups.values() if len(v)>1)})")

# ---- pass2: 转换 + 装载 + 收集 int8 ----
OUTQ, nq = {}, 0
t0 = time.time()
with safe_open(f"{ENC_DIR}/model.safetensors", framework="pt") as f:
    for n, m in list(enc.named_modules()):
        if not isinstance(m, nn.Linear) or not n.startswith("model.language_model"):
            continue
        W = f.get_tensor(n + ".weight")
        if W.dtype == torch.float8_e4m3fn:
            W = (W.to(torch.float32) * f.get_tensor(n + ".weight_scale")).to(torch.bfloat16)
        bias = None
        try:
            bias = f.get_tensor(n + ".bias")
        except Exception:
            pass
        ssm = SSM[group_of(n)]
        sinv = (1.0 / ssm).float()
        Wr = rot_w(W.float(), ssm, W.shape[1])
        s = Wr.abs().amax(dim=1) / 127.0
        q = torch.round(Wr / s.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        wq = torch._make_per_channel_quantized_tensor(q, s, torch.zeros_like(s, dtype=torch.int32), 0)
        OUTQ[n] = {"int8": q.cpu().contiguous(), "scale": s.cpu().contiguous(), "sinv": sinv.cpu().contiguous()}
        if bias is not None:
            OUTQ[n]["bias"] = bias.to(torch.bfloat16).cpu().contiguous()
        parts = n.split("."); parent = enc
        for x_ in parts[:-1]:
            parent = getattr(parent, x_)
        setattr(parent, parts[-1], RLinear(m.in_features, m.out_features, wq, bias, sinv))
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
print(f"pass2 转换 {nq} 层 {time.time()-t0:.0f}s")
for m in enc.modules():
    inv = getattr(m, "inv_freq", None)
    if inv is not None and inv.is_meta:
        dim = inv.shape[-1] * 2
        m.inv_freq = 1.0 / (5000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
enc = enc.eval()

# ---- 验证 cos (在线蝶形路径) ----
coss = []
with torch.no_grad():
    for p in PROMPTS:
        ids = tok(TPL.format(p), return_tensors="pt")["input_ids"]
        out = enc(ids, output_hidden_states=True, use_cache=False)
        ha, hb = out.hidden_states[-1][0].float(), base[p]
        coss.append(torch.nn.functional.cosine_similarity(
            ha.flatten().unsqueeze(0), hb.flatten().unsqueeze(0)).item())
        del out
print("cos:", " ".join(f"{c:.6f}" for c in coss), f"min={min(coss):.6f} gate={COS_GATE}")

if min(coss) >= COS_GATE:
    out = {}
    for n, d in OUTQ.items():
        for k, v in d.items():
            out[n + "." + k] = v
    save_file(out, OUT)
    print("SAVED", OUT, len(out))
else:
    print("COS_UNDERSHOT, 不落盘")
print("SQROT2_CONV_DONE")
