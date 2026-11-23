# 迷你 SpinQuant: 逐层学习组间旋转 Hq (Cayley 正交 + 完整 W8A8 量化 MSE + STE), GPU
from qwen21dit.paths import MODEL_DIR
import sys, json, time, gc, torch
import torch.nn as nn
from safetensors.torch import safe_open, save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

ENC_DIR = str(MODEL_DIR / "fp8/qwen3vl8b_fp8")
XS = str(MODEL_DIR / "int8_qwen3vl8b.xsamples.safetensors")
OUT = str(MODEL_DIR / "int8_qwen3vl8b.rotlearn.safetensors")
N0, SEED = 256, 20261122
STEPS, LR = 120, 3e-3
torch.manual_seed(SEED)

def reg_hadamard_4():
    return torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                         [1., -1., 1., 1.], [-1., 1., 1., 1.]])

def kron(a, b):
    return torch.einsum("ij,kl->ikjl", a, b).reshape(a.shape[0]*b.shape[0], a.shape[1]*b.shape[1])

h = reg_hadamard_4()
for _ in range(3):
    h = kron(h, reg_hadamard_4())
R256 = (torch.diag(torch.randn(256).sign()) @ (h / 16.0)).cuda()   # 与 convert_rht_verify 同构

def rot2_simple(x, Hq):
    K = x.shape[-1]
    NG = K // N0
    shp = x.shape
    xf = x.reshape(-1, K)
    x1 = torch.einsum("bgn,hg->bhn", xf.view(-1, NG, N0), Hq)
    return (x1.reshape(-1, N0) @ R256).reshape(shp)

def ste_round(x):
    return x + (torch.round(x) - x).detach()

def qsim(Xr, Wr):   # 完整 W8A8 量化模拟 (STE)
    sw = Wr.abs().amax(dim=1, keepdim=True) / 127.0
    wq = ste_round(Wr / sw) * sw
    sa = Xr.abs().amax(dim=-1, keepdim=True) / 127.0
    xq = ste_round(Xr / sa) * sa
    return xq @ wq.T

xs = {}
with safe_open(XS, framework="pt") as f:
    for k in f.keys():
        xs[k] = f.get_tensor(k)

cfg = json.load(open(f"{ENC_DIR}/config.json")); cfg.pop("quantization_config", None)
old = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
with torch.device("meta"):
    enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg))
torch.set_default_dtype(old)

hqs, t0 = {}, time.time()
with safe_open(f"{ENC_DIR}/model.safetensors", framework="pt") as f:
    lins = [(n, m) for n, m in enc.named_modules()
            if isinstance(m, nn.Linear) and n.startswith("model.language_model") and (n + ".weight") in f.keys()]
    for i, (n, m) in enumerate(lins):
        X = xs[n].cuda()                       # [96,K]
        W = f.get_tensor(n + ".weight")
        if W.dtype == torch.float8_e4m3fn:
            W = (W.to(torch.float32) * f.get_tensor(n + ".weight_scale")).float()
        W = W.cuda()
        K = W.shape[1]
        NG = K // N0
        Y_ref = X @ W.T
        g = torch.Generator(device="cpu").manual_seed(SEED + i)
        M = torch.randn(NG, NG, generator=g) * 0.1
        A = (M - M.T).cuda().requires_grad_(True)
        opt = torch.optim.Adam([A], lr=LR)
        best = (1e9, None)
        for step in range(STEPS):
            Hq = torch.linalg.solve(torch.eye(NG, device="cuda") + A,
                                    torch.eye(NG, device="cuda") - A)     # Cayley 正交
            Xr = rot2_simple(X, Hq)
            Wr = rot2_simple(W, Hq)
            loss = ((qsim(Xr, Wr) - Y_ref) ** 2).mean() / (Y_ref ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best[0]:
                best = (loss.item(), Hq.detach().cpu())
        hqs[n + ".Hq"] = best[1].contiguous()
        del X, W, Y_ref
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(lins)} 层, 累计 {time.time()-t0:.0f}s, 最新 loss={best[0]:.5f}")

save_file(hqs, OUT)
print(f"OK {OUT}: {len(hqs)} 个 Hq, 总耗时 {time.time()-t0:.0f}s")
print("ROTLEARN_DONE")
