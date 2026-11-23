# 定版转换: SmoothQuant α=0.75 W8A8 权重落盘 (int8 + scale + sinv + bf16 层)
from qwen21dit.paths import MODEL_DIR
import json, torch, os
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from safetensors.torch import safe_open, save_file

ENC_DIR = str(MODEL_DIR / "fp8/qwen3vl8b_fp8")
CAL = str(MODEL_DIR / "int8_qwen3vl8b.calib.safetensors")
OUT = str(MODEL_DIR / "int8_qwen3vl8b_sq.safetensors")
ALPHA = 0.75

cfg = json.load(open(f"{ENC_DIR}/config.json")); cfg.pop("quantization_config", None)
old = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
with torch.device("meta"):
    enc = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**cfg))
torch.set_default_dtype(old)

cal = {}
with safe_open(CAL, framework="pt") as f:
    for k in f.keys():
        cal[k] = f.get_tensor(k)

out_sd, meta = {}, {"sq": [], "bf16": []}
with safe_open(f"{ENC_DIR}/model.safetensors", framework="pt") as f:
    for n, m in list(enc.named_modules()):
        if not isinstance(m, torch.nn.Linear) or n not in cal:
            continue
        k = n + ".weight"
        W = f.get_tensor(k)
        if W.dtype == torch.float8_e4m3fn:
            W = (W.to(torch.float32) * f.get_tensor(k + "_scale")).to(torch.bfloat16)
        Wf = W.float()
        maxW = Wf.abs().amax(dim=0).clamp(min=1e-8)
        maxX = cal[n].clamp(min=1e-8)
        s = (maxX ** ALPHA) / (maxW ** (1 - ALPHA))
        W2 = Wf * s.unsqueeze(0)
        scale_row = W2.abs().amax(dim=1) / 127.0
        q = torch.round(W2 / scale_row.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        out_sd[k + ".int8"] = q.contiguous()
        out_sd[k + ".scale"] = scale_row.to(torch.float32).contiguous()
        out_sd[k + ".sinv"] = (1.0 / s).to(torch.float32).contiguous()
        try:
            out_sd[n + ".bias.bf16"] = f.get_tensor(n + ".bias").to(torch.bfloat16).contiguous()
        except Exception:
            pass
        meta["sq"].append(n)
        del W, Wf, W2, q
    # 非 SQ Linear 覆盖的其余层: bf16 (embedding/norm/visual 塔 skip)
    keys = set(f.keys())
    for name, module in enc.named_modules():
        for pname in [pn for pn, _ in module.named_parameters(recurse=False)]:
            full = f"{name}.{pname}" if name else pname
            if full in keys and (full + "_scale") not in keys:
                t = f.get_tensor(full)
                if t.dtype == torch.float8_e4m3fn:
                    continue
                out_sd[full + ".bf16"] = t.to(torch.bfloat16).contiguous()
                meta["bf16"].append(full)

save_file(out_sd, OUT, metadata={"format": "pt", "alpha": str(ALPHA)})
json.dump(meta, open(OUT.replace(".safetensors", ".manifest.json"), "w"))
print(f"OK {OUT} {os.path.getsize(OUT)/2**30:.2f}GB  sq={len(meta['sq'])} bf16={len(meta['bf16'])}")
