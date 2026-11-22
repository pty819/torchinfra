# VAE decode 浪费画像：小图无 tiling 基线 + profiler 热点 + patch A/B
import sys, time, json
import torch
import torch.nn.functional as F

VAE_CFG = "~/models/qwen21/fp8/vae21_config.json"
VAE_W = "~/models/qwen21/qwen_image_2.1_vae_diffusers.safetensors"

from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
cfgd = json.load(open(VAE_CFG))
vae = AutoencoderKLQwenImage21(**cfgd).eval()
# 只留 decoder，省显存
for a in ("encoder", "quant"):
    if hasattr(vae, a):
        delattr(vae, a)
vae = vae.to("cuda", torch.bfloat16)
sd = vae.state_dict()
from safetensors.torch import load_file
raw = load_file(VAE_W)
remap = {}
for k, v in raw.items():
    kk = k.replace("decoder.", "decoder.").replace("encoder.", "encoder.")
    if kk in sd and sd[kk].shape == v.shape:
        if kk.startswith("decoder."): remap[kk] = v
vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in remap.items()}, strict=False)
print(f"loaded {len(remap)}/{len([k for k in sd if k.startswith(chr(100)+chr(101)+chr(99))])} dec tensors, vram={torch.cuda.memory_allocated()/2**30:.2f}GB")

H = W = 11  # → 176² 图, 显存最小化
lat = torch.randn(1, 64, 1, H, W, device="cuda", dtype=torch.bfloat16) * 3
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

with torch.no_grad():
    out_ref = vae.decode(lat, return_dict=False)[0]
print("out:", out_ref.shape, out_ref.dtype)

t_full = bench(lambda: vae.decode(lat, return_dict=False))
print(f"\n[基线] 小图(336²) 无tiling decode: {t_full*1000:.1f}ms  "
      f"(外推按 H² 缩放)")

# ---- profiler 热点 ----
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        vae.decode(lat, return_dict=False)
evs = prof.key_averages()
rows = sorted(evs, key=lambda e: -e.self_device_time_total)[:14]
print(f"\n{'kernel':<58}{'self_ms':>9}{'cnt':>6}")
for e in rows:
    print(f"{e.key[:58]:<58}{e.self_device_time_total/1000:>9.2f}{e.count:>6}")
tot = sum(e.self_device_time_total for e in evs) / 1000
print(f"{'TOTAL':<58}{tot:>9.2f}")
print("PROFILE_DONE")
