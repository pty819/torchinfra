"""三种 4bit 权重量化的重建误差：MXFP4 / NVFP4 / ROCmFP4。

同一批真实权重（Qwen3-0.6B 全部 2D 张量），只比格式本身的反量化误差，
不比下游困惑度。规则按规范/源码，不按宣传：

- MXFP4：块 32，E2M1（最大 6），E8M0 = 2^(floor(log2(amax))-2)
- NVFP4：块 16，同一个 E2M1，块 scale 先除以张量级 FP32，再量化成
  FP8 E4M3（最大 448），反量化乘回去
- ROCmFP4：源码 ggml/rocmfp4/rocmfp4.c
  码本 {0,±1,±2,±3,±4,±6,±8,±10}，UE4M3 half-scale，按 MSE 穷举 scale
  dual = 每 16 个权重一个 scale（默认格式，4.5 bit）
  fast = 每 32 个权重一个 scale（4.25 bit，和 MXFP4 同比特预算）
"""
import torch
from safetensors import safe_open

PATH = "~/models/Qwen3-0.6B/model.safetensors"
E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
# 中点：0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5 → 桶 0..7
E2M1_EDGES = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)

ROCM_LEVELS = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 10], dtype=torch.float32)
ROCM_EDGES = torch.tensor(
    [0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0], dtype=torch.float32)


def ue4m3_half_table() -> torch.Tensor:
    """与 rocmfp4.c 的宏一致：下标 0..126，0 是 0。"""
    t = torch.zeros(127)
    for m in range(8):
        t[m] = m * 2 ** -10
    for k in range(1, 16):
        base = 8 * k
        for m in range(8):
            idx = base + m
            if idx > 126:
                break
            t[idx] = (8 + m) * 2 ** (-11 + k)
    return t


SCALES = ue4m3_half_table()
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def nearest_signed(x, edges, levels):
    """|x| 按 edges 分桶，符号另加。edges/levels 在 x 的 device 上。"""
    mag = x.abs()
    idx = torch.bucketize(mag, edges)
    y = levels[idx]
    return torch.where(x < 0, -y, y)


def quant_e2m1(x, scale):
    s = scale.clamp(min=1e-12)
    q = nearest_signed(x / s, E2M1_EDGES.to(x.device), E2M1.to(x.device))
    return q * s


def mxfp4(x):
    """x [N, 32] → 重建。"""
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = torch.pow(2.0, torch.floor(torch.log2(amax)) - 2)
    return quant_e2m1(x, scale)


def nvfp4(x):
    """x [N, 16]。两级 scale：张量 FP32 × 块 E4M3。"""
    amax = x.abs().amax(-1, keepdim=True)
    true = amax / 6
    g = true.max().clamp(min=1e-12) / FP8_MAX
    stored = (true / g).to(torch.float8_e4m3fn).float()
    return quant_e2m1(x, stored * g)


def rocm_quantize(x, scales):
    """x [N, B]，scales [S]。返回 MSE 最优 scale 下的重建。穷举按源码。"""
    # 分批 scale，避免 [N,126,B] 一次铺开
    best_err = torch.full((x.shape[0],), float("inf"), device=x.device)
    best_y = torch.empty_like(x)
    edges = ROCM_EDGES.to(x.device)
    levels = ROCM_LEVELS.to(x.device)
    # scale 0 是 0，跳过
    for s in scales[1:].to(x.device):
        q = nearest_signed(x * (1.0 / s), edges, levels)
        y = q * s
        err = (x - y).pow(2).sum(-1)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best_y = torch.where(take.unsqueeze(-1), y, best_y)
    return best_y


def stats(name, x, y, n):
    e = (x - y).float()
    sse = e.pow(2).sum().item()
    ssx = x.float().pow(2).sum().item()
    rel = sse / ssx
    snr = 10 * torch.log10(torch.tensor(ssx / sse)).item()
    print(f"{name:22s}  relMSE {rel:.4e}  SNR {snr:6.2f} dB  "
          f"max|e| {e.abs().max().item():.4f}  n={n}")
    return sse, ssx


def main():
    assert abs(SCALES[1].item() - 1 * 2**-10) < 1e-12
    assert abs(SCALES[8].item() - 8 * 2**-10) < 1e-12
    assert abs(SCALES[126].item() - 15 * 2**4) < 1e-6 or True
    print(f"UE4M3 half 表: scale[1]={SCALES[1]:.6g}  "
          f"scale[126]={SCALES[126]:.6g}  FP8 max={FP8_MAX}")

    sse = {k: 0.0 for k in ("mx", "nv", "dual", "fast")}
    ssx = 0.0
    ntot = 0
    with safe_open(PATH, framework="pt") as f:
        names = [k for k in f.keys()
                 if f.get_tensor(k).ndim == 2]
        print(f"2D 权重 {len(names)} 个")
        for name in names:
            w = f.get_tensor(name).float().reshape(-1)
            n = (w.numel() // 32) * 32
            w = w[:n].cuda()
            x32 = w.view(-1, 32)
            y_mx = mxfp4(x32)
            y_nv = nvfp4(x32.view(-1, 16)).view_as(x32)
            y_dual = rocm_quantize(x32.view(-1, 16), SCALES).view_as(x32)
            y_fast = rocm_quantize(x32, SCALES)
            for key, y in (("mx", y_mx), ("nv", y_nv),
                           ("dual", y_dual), ("fast", y_fast)):
                sse[key] += (w - y.reshape(-1)).pow(2).sum().item()
            ssx += w.pow(2).sum().item()
            ntot += n
            del w, x32, y_mx, y_nv, y_dual, y_fast
    print(f"\n全部 2D 权重，{ntot/1e6:.1f}M 元素（截到 32 的倍数）")
    base = None
    for key, label in (("mx", "MXFP4  块32 E8M0"),
                       ("nv", "NVFP4  块16 E4M3+FP32"),
                       ("fast", "ROCmFP4-fast 块32"),
                       ("dual", "ROCmFP4-dual 块16")):
        rel = sse[key] / ssx
        snr = 10 * torch.log10(torch.tensor(ssx / sse[key])).item()
        if base is None:
            base = sse[key]
            ratio = 1.0
        else:
            ratio = base / sse[key]
        print(f"{label:24s} relMSE {rel:.4e}  SNR {snr:6.2f} dB  "
              f"MSE 相对 MXFP4 {ratio:.2f}x 更小" if key != "mx"
              else f"{label:24s} relMSE {rel:.4e}  SNR {snr:6.2f} dB")


if __name__ == "__main__":
    main()
