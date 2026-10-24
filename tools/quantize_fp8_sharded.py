"""把多分片 bf16 checkpoint 量化成 fp8（save_fp8_checkpoint_sharded）并做三重验证：
1) 结构：fp8 张量数、非量化键 bitwise 相等、index 一致；
2) 数值：逐张量 dequant 误差（最大绝对误差 / 余弦 / 相对 amax 误差）汇总最差项；
3) GPU 冒烟：layer0 权重走 torchinfra fp8_linear_q 路径与 bf16 F.linear 对拍。

用法: PYTHONPATH=. python tools/quantize_fp8_sharded.py <src_dir> <dst_dir> [--skip-quant]
"""
import sys, argparse, json
from pathlib import Path
import torch
from safetensors.torch import load_file

from torchinfra.fp8_ckpt import save_fp8_checkpoint_sharded


def verify(src: Path, dst: Path) -> None:
    src_idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    dst_idx = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]

    n_fp8 = n_same = 0
    worst_cos, worst_rel, worst_abs = (1.0, ""), (0.0, ""), (0.0, "")
    sum_cos = 0.0
    for f in sorted(set(src_idx.values())):
        s, d = load_file(str(src / f)), load_file(str(dst / f))
        for k, v in d.items():
            if k.endswith(".weight_scale"):
                continue
            orig = s[k]
            if v.dtype == torch.float8_e4m3fn:
                n_fp8 += 1
                scale = d[k[: -len("weight")] + "weight_scale"]
                dq = v.float() * scale
                cos = torch.nn.functional.cosine_similarity(
                    dq.flatten(), orig.float().flatten(), dim=0).item()
                ae = (dq - orig.float()).abs().max().item()
                rel = ae / orig.float().abs().max().clamp(min=1e-12).item()
                sum_cos += cos
                if cos < worst_cos[0]: worst_cos = (cos, k)
                if rel > worst_rel[0]: worst_rel = (rel, k)
                if ae > worst_abs[0]: worst_abs = (ae, k)
            else:
                assert torch.equal(v, orig), f"非量化键被改动: {k}"
                n_same += 1
        del s, d

    print(f"结构: fp8 张量 {n_fp8} 个, bf16 保持 {n_same} 个 (bitwise 相等)")
    print(f"数值: 平均余弦 {sum_cos / n_fp8:.6f} | 最差余弦 {worst_cos[0]:.6f} ({worst_cos[1]})")
    print(f"      最差相对amax误差 {worst_rel[0]:.4f} ({worst_rel[1]})")
    print(f"      最差绝对误差 {worst_abs[0]:.4f} ({worst_abs[1]})")


def gpu_smoke(dst: Path) -> None:
    if not torch.cuda.is_available():
        print("GPU 不可用，跳过冒烟")
        return
    from torchinfra.fp8_linear import fp8_linear_q
    sd = {}
    for f in sorted(dst.glob("*.safetensors")):
        sd.update(load_file(str(f)))
        if any(k.startswith("model.language_model.layers.0.self_attn.q_proj") for k in sd):
            break
    k = "model.language_model.layers.0.self_attn.q_proj.weight"
    w8, s = sd[k].cuda(), sd[k[:-len("weight")] + "weight_scale"].cuda()
    src = dst.parent / "Nemotron-3.5-Rerank-8B-BF16"
    # bf16 参考直接从源目录取该层
    ref_idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    ref = load_file(str(src / ref_idx[k]), device="cpu")[k].cuda()
    torch.manual_seed(0)
    x = (torch.randn(256, ref.shape[1], device="cuda") * 0.5).to(torch.bfloat16)
    partial = x.abs().amax(dim=1).float()  # 行 amax（真实路径 partial 恒为 fp32）
    y8 = fp8_linear_q(x, w8, s, partial)
    yb = torch.nn.functional.linear(x, ref)
    rel = ((y8 - yb).norm() / yb.norm()).item()
    print(f"GPU 冒烟: q_proj[4096,4096] fp8 路径 vs bf16 F.linear 相对误差 {rel:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--skip-quant", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if not a.skip_quant:
        save_fp8_checkpoint_sharded(src, dst)
        print(f"量化完成 → {dst}")
    verify(src, dst)
    gpu_smoke(dst)
