# tiling 税量化（176² 小图版）：wall(块数) 拟合 per-tile 固定开销
import time, json
import torch

VAE_CFG = "~/models/qwen21/fp8/vae21_config.json"
VAE_W = "~/models/qwen21/qwen_image_2.1_vae_diffusers.safetensors"
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from safetensors.torch import load_file

cfgd = json.load(open(VAE_CFG))
vae = AutoencoderKLQwenImage21(**cfgd).eval()
for a in ("encoder", "quant"):
    if hasattr(vae, a):
        delattr(vae, a)
vae = vae.to("cuda", torch.bfloat16)
sd = vae.state_dict()
raw = load_file(VAE_W)
vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in raw.items()
                     if k in sd and sd[k].shape == v.shape}, strict=False)

H = W = 11  # → 176²
g = torch.Generator(device='cuda').manual_seed(7)
lat = torch.randn(1, 64, 1, H, W, device='cuda', dtype=torch.bfloat16, generator=g) * 3
mean = torch.tensor(cfgd["latents_mean"], device="cuda", dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
std = torch.tensor(cfgd["latents_std"], device="cuda", dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
lat = lat * std + mean

def bench(fn, n=5, warm=2):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

def ntiles(size, tile, stride):
    return (1 + max(0, -(-(size - tile) // stride))) ** 2

with torch.no_grad():
    vae.disable_tiling()
    t_full = bench(lambda: vae.decode(lat, return_dict=False))
    o_full = vae.decode(lat, return_dict=False)[0]
    print(f"无tiling(1块):   {t_full*1000:7.1f}ms")

    for tile, stride in ((96, 72), (64, 48), (48, 32)):
        vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                          tile_sample_stride_height=stride, tile_sample_stride_width=stride)
        n = ntiles(176, tile, stride)
        t = bench(lambda: vae.decode(lat, return_dict=False))
        o = vae.decode(lat, return_dict=False)[0]
        d = (o.float() - o_full.float()).abs().max().item()
        print(f"tile{tile}/{stride} ({n:2d}块): {t*1000:7.1f}ms  Δ={d:.4f}  "
              f"每块固定开销≈{(t - t_full) / (n - 1) * 1000 if n > 1 else 0:5.1f}ms")

print("TAX_DONE")
