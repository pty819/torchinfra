# 1344² 全尺寸 tiling 税 + 正确性对照（worker 已卸载，显存 11.9GB 可用）
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

H = W = 84  # → 1344²
g = torch.Generator(device='cuda').manual_seed(7)
lat = torch.randn(1, 64, 1, H, W, device='cuda', dtype=torch.bfloat16, generator=g) * 3
mean = torch.tensor(cfgd["latents_mean"], device="cuda", dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
std = torch.tensor(cfgd["latents_std"], device="cuda", dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
lat = lat * std + mean

def bench(fn, n=3, warm=1):
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
    o_full = None
    for tile, stride in ((256, 192), (512, 384), (640, 512), (768, 576)):
        vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                          tile_sample_stride_height=stride, tile_sample_stride_width=stride)
        n = ntiles(1344, tile, stride)
        torch.cuda.reset_peak_memory_stats()
        t = bench(lambda: vae.decode(lat, return_dict=False))
        peak = torch.cuda.max_memory_allocated() / 2**30
        o = vae.decode(lat, return_dict=False)[0].float().cpu()
        if o_full is not None:
            d = (o - o_full).abs().max().item()
        else:
            d = float("nan")
        print(f"tile{tile}/{stride} ({n:2d}块): {t:6.2f}s  peak={peak:.2f}GB  Δ={d:.4f}")

    try:
        vae.disable_tiling()
        torch.cuda.reset_peak_memory_stats()
        t_full = bench(lambda: vae.decode(lat, return_dict=False))
        peak_full = torch.cuda.max_memory_allocated() / 2**30
        o_full = vae.decode(lat, return_dict=False)[0].float().cpu()
        print(f"无tiling:     {t_full:6.2f}s  peak={peak_full:.2f}GB")
    except torch.OutOfMemoryError:
        print("无tiling: OOM 跳过")
print("FULL_TAX_DONE")
