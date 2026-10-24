"""把 bf16 checkpoint 收成混合精度：Linear 权重 E4M3 + 每矩阵一个 scale，其余保持 bf16。

E4M3 只能表示到 ±448，所以每个权重矩阵要存一个 fp32 scale，
weight_q = round(weight / scale)。这是存盘格式，不是运行时再搜 scale。
embedding、RMSNorm、q/k norm、projector 不动。
"""
from pathlib import Path
import json
import shutil
import torch
from safetensors.torch import load_file, save_file

from torchinfra.fp8_linear import FP8_MAX

_QUANT_SUFFIX = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)
_COPY = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt",
)


def is_fp8_checkpoint(model_dir: str | Path) -> bool:
    cfg = Path(model_dir) / "config.json"
    if not cfg.is_file():
        return False
    raw = json.loads(cfg.read_text())
    return raw.get("quantization_config", {}).get("quant_method") == "fp8"


def _quantize_tensor(w: torch.Tensor):
    s = (w.float().abs().amax() / FP8_MAX).clamp(min=1e-12).reshape(1).float()
    q = (w.float() / s).to(torch.float8_e4m3fn).contiguous()
    return q, s


def _put(out: dict, key: str, w: torch.Tensor, scale: torch.Tensor | None = None):
    """key 以 .weight 结尾。scale 给定时不再单独算（用于 q/k/v 共用一个 scale）。"""
    if scale is None:
        q, scale = _quantize_tensor(w)
    else:
        q = (w.float() / scale).to(torch.float8_e4m3fn).contiguous()
    out[key] = q
    out[key[: -len("weight")] + "weight_scale"] = scale.reshape(1).contiguous().clone()


def save_fp8_checkpoint(src: str | Path, dst: str | Path) -> Path:
    """src 是原始 bf16 目录，dst 写成混合 fp8 checkpoint。

    Q/K/V 共用一个 scale，gate/up 共用一个 scale，这样加载后可以拼回一次 GEMM，
    而且拼回去的字节就是文件里的字节，不会再量化第二次。
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(src / "model.safetensors"))
    out = {}
    n_fp8 = 0
    layers = sorted({k.split(".self_attn.")[0] for k in sd if ".self_attn.q_proj.weight" in k})
    for layer in layers:
        a = f"{layer}.self_attn"
        fused_qkv = torch.cat([sd[f"{a}.q_proj.weight"], sd[f"{a}.k_proj.weight"],
                               sd[f"{a}.v_proj.weight"]], dim=0)
        q, s = _quantize_tensor(fused_qkv)
        nq, nk = sd[f"{a}.q_proj.weight"].shape[0], sd[f"{a}.k_proj.weight"].shape[0]
        _put(out, f"{a}.q_proj.weight", sd[f"{a}.q_proj.weight"], s)
        _put(out, f"{a}.k_proj.weight", sd[f"{a}.k_proj.weight"], s)
        _put(out, f"{a}.v_proj.weight", sd[f"{a}.v_proj.weight"], s)
        # 上面三次各自又量化了一次。改成直接切已经量化好的 fused，避免和 s 再除一次。
        out[f"{a}.q_proj.weight"] = q[:nq].contiguous().clone()
        out[f"{a}.k_proj.weight"] = q[nq:nq + nk].contiguous().clone()
        out[f"{a}.v_proj.weight"] = q[nq + nk:].contiguous().clone()
        _put(out, f"{a}.o_proj.weight", sd[f"{a}.o_proj.weight"])
        m = f"{layer}.mlp"
        fused_gu = torch.cat([sd[f"{m}.gate_proj.weight"], sd[f"{m}.up_proj.weight"]], dim=0)
        gq, gs = _quantize_tensor(fused_gu)
        ng = sd[f"{m}.gate_proj.weight"].shape[0]
        _put(out, f"{m}.gate_proj.weight", sd[f"{m}.gate_proj.weight"], gs)
        _put(out, f"{m}.up_proj.weight", sd[f"{m}.up_proj.weight"], gs)
        out[f"{m}.gate_proj.weight"] = gq[:ng].contiguous().clone()
        out[f"{m}.up_proj.weight"] = gq[ng:].contiguous().clone()
        _put(out, f"{m}.down_proj.weight", sd[f"{m}.down_proj.weight"])
        n_fp8 += 7
    for k, v in sd.items():
        if k not in out and not k.endswith(_QUANT_SUFFIX):
            out[k] = v
    save_file(out, str(dst / "model.safetensors"))
    raw = json.loads((src / "config.json").read_text())
    raw["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "float8_e4m3fn",
        "weight_scale": "tensor",
        "activation_scheme": "dynamic",
        "ignored": ["embed_tokens", "lm_head", "projector", "norm",
                    "q_norm", "k_norm"],
        "n_fp8_tensors": n_fp8,
    }
    (dst / "config.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    for name in _COPY:
        if name == "config.json":
            continue
        f = src / name
        if f.is_file():
            shutil.copy2(f, dst / name)
    return dst


def save_fp8_checkpoint_sharded(src: str | Path, dst: str | Path) -> Path:
    """多分片 bf16 checkpoint 的通用版：自动发现文本塔层前缀（兼容
    `model.layers.N` / `model.language_model.layers.N`），7 个 Linear 量化成
    E4M3（QKV 共 scale、gate/up 共 scale，运行时拼回一次 GEMM 的字节就是盘上的
    字节），视觉塔/projector/score head/embedding/norm 保持 bf16。

    输出镜像原分片布局（scale 跟随对应 .weight 所在分片），index 的
    total_size 按输出重算。内存策略：全量加载后每量化一个 Linear 就地
    pop 掉 bf16 源，峰值 ≈ 原模型 + 一份 fp8。
    """
    import re

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wm: dict[str, str] = idx["weight_map"]

    sd: dict[str, torch.Tensor] = {}
    for f in sorted(set(wm.values())):
        sd.update(load_file(str(src / f)))

    layers = sorted(
        {m.group(1) for k in sd
         for m in [re.match(r"(.+\.layers\.\d+)\.self_attn\.q_proj\.weight$", k)] if m},
        key=lambda p: int(p.rsplit(".", 1)[1]),
    )
    if not layers:
        raise ValueError("未发现任何 `<prefix>.layers.N.self_attn.q_proj.weight`，不是已知布局")

    n_fp8 = 0
    for layer in layers:
        a, m = f"{layer}.self_attn", f"{layer}.mlp"
        shapes = {n: sd[f"{a}.{n}_proj.weight"].shape for n in "qkv"}
        fused = torch.cat([sd.pop(f"{a}.{n}_proj.weight") for n in "qkv"], dim=0)
        q, s = _quantize_tensor(fused)
        del fused
        nq = shapes["q"][0]
        nk = shapes["k"][0]
        for i, n in enumerate("qkv"):
            beg = 0 if n == "q" else (nq if n == "k" else nq + nk)
            end = beg + shapes[n][0]
            sd[f"{a}.{n}_proj.weight"] = q[beg:end].clone()
            sd[f"{a}.{n}_proj.weight_scale"] = s.clone()
        for name in ("o_proj", "down_proj"):
            w = sd.pop(f"{a if name == 'o_proj' else m}.{name}.weight")
            qs, ss = _quantize_tensor(w)
            del w
            sd[f"{a if name == 'o_proj' else m}.{name}.weight"] = qs
            sd[f"{a if name == 'o_proj' else m}.{name}.weight_scale"] = ss
        gshapes = (sd[f"{m}.gate_proj.weight"].shape[0],
                   sd[f"{m}.up_proj.weight"].shape[0])
        fused = torch.cat([sd.pop(f"{m}.gate_proj.weight"), sd.pop(f"{m}.up_proj.weight")], dim=0)
        gq, gs = _quantize_tensor(fused)
        del fused
        sd[f"{m}.gate_proj.weight"] = gq[: gshapes[0]].clone()
        sd[f"{m}.gate_proj.weight_scale"] = gs.clone()
        sd[f"{m}.up_proj.weight"] = gq[gshapes[0]:].clone()
        sd[f"{m}.up_proj.weight_scale"] = gs.clone()
        n_fp8 += 7
    del layers

    # 输出镜像原分片：scale 跟随 .weight 所在分片
    out_shards: dict[str, dict[str, torch.Tensor]] = {f: {} for f in set(wm.values())}
    for key, f in wm.items():
        out_shards[f][key] = sd.pop(key)
        if key.endswith(".weight") and (key[: -len("weight")] + "weight_scale") in sd:
            out_shards[f][key[: -len("weight")] + "weight_scale"] = sd.pop(
                key[: -len("weight")] + "weight_scale")
    assert not sd, f"有张量没归入任何分片: {list(sd)[:5]}"

    total = 0
    for f, tensors in sorted(out_shards.items()):
        save_file(tensors, str(dst / f))
        total += sum(t.numel() * t.element_size() for t in tensors.values())

    new_idx = {"metadata": {"total_size": total}, "weight_map": {}}
    for f, tensors in out_shards.items():
        for key in tensors:
            new_idx["weight_map"][key] = f
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(new_idx, indent=2) + "\n")

    raw = json.loads((src / "config.json").read_text())
    raw["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "float8_e4m3fn",
        "weight_scale": "tensor",
        "activation_scheme": "dynamic",
        "ignored": ["embed_tokens", "score", "projector", "norm",
                    "q_norm", "k_norm", "vision_tower"],
        "n_fp8_tensors": n_fp8,
    }
    (dst / "config.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")

    # 其余文件（model.py / chat_template / tokenizer / sentence-transformers 配置等）全量复制
    for p in src.iterdir():
        if p.is_file() and p.suffix not in (".safetensors",) and p.name != "config.json":
            shutil.copy2(p, dst / p.name)
    for d in src.iterdir():
        if d.is_dir():
            shutil.copytree(d, dst / d.name, dirs_exist_ok=True)
    return dst


def load_fp8_checkpoint(model_dir: str | Path, device: str = "cuda"):
    """返回 (bf16 状态字典, fp8 权重)。

    fp8 字典的值是 (e4m3 张量, scale [1])，key 与原始 `.weight` 相同。
    bf16 字典里的 Linear 只是用 scale 还原出来给 load_state_dict 占位，
    真正的 GEMM 用 fp8 那份，不再量化第二次。
    """
    model_dir = Path(model_dir)
    sd = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    bf16, fp8 = {}, {}
    for k, v in sd.items():
        if k.endswith(".weight_scale"):
            continue
        if v.dtype == torch.float8_e4m3fn:
            scale = sd[k[: -len("weight")] + "weight_scale"].to(device=device).reshape(1)
            q = v.to(device=device).contiguous()
            fp8[k] = (q, scale)
            bf16[k] = (q.float() * scale).to(torch.bfloat16)
        else:
            bf16[k] = v.to(device=device, dtype=torch.bfloat16)
    if "lm_head.weight" not in bf16:
        bf16["lm_head.weight"] = bf16["model.embed_tokens.weight"]
    return bf16, fp8
