from qwen21dit.paths import MODEL_DIR
import json
from pathlib import Path
from safetensors import safe_open
def inspect(path, out):
    r = {"path": path, "tensors": {}}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            t = f.get_slice(k)
            r["tensors"][k] = [list(t.get_shape()), str(t.get_dtype())]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(out, "w"), indent=1)
    print(out, len(r["tensors"]), "tensors")
if __name__ == "__main__":
    inspect(str(MODEL_DIR / "fp8/qwen21_fp8_e4m3fn.safetensors"), "reports/dit_report.json")
    inspect(str(MODEL_DIR / "fp8/omni_lora_v0.1_bf16.safetensors"), "reports/omni_report.json")
    inspect(str(MODEL_DIR / "loras/nsfw_alpacas_v2.safetensors"), "reports/alpacas_report.json")
