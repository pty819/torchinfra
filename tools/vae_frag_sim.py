import os
# 复现 worker 环境税：占住 ~7.5GB 杂乱块后跑同样的 tiling decode
import time, json
import torch

VAE_CFG = os.path.expanduser("~/models/qwen21/fp8/vae21_config.json")
VAE_W = os.path.expanduser("~/models/qwen21/qwen_image_2.1_vae_diffusers.safetensors")
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

H = W = 84
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

with torch.no_grad():
    vae.enable_tiling()  # 生产同款默认参数

    # 先占用再冷启动 decode (worker 顺序!)
    hog0 = []
    for sh in [(1 << 30)] * 6 + [(153_000_000)] * 2 + [(37_000_000)] * 4:
        try:
            hog0.append(torch.empty(sh, device="cuda", dtype=torch.uint8))
        except torch.OutOfMemoryError:
            break
    print(f"预占用: alloc={torch.cuda.memory_allocated()/2**30:.2f}GB")
    t_first = bench(lambda: vae.decode(lat, return_dict=False), n=3, warm=1)
    print(f"压力下冷启动 49块: {t_first:.2f}s")
    for h in hog0: h = None
    del hog0
    torch.cuda.empty_cache()
    t_relaxed = bench(lambda: vae.decode(lat, return_dict=False))
    print(f"释放后(算法已缓存) 49块: {t_relaxed:.2f}s")
    raise SystemExit(0)

    # 模拟 worker: DiT fp8 权重 + 激活痕迹的杂乱占用 ~7.3GB
    hog = []
    shapes = [(1 << 30)] * 6 + [(153_000_000)] * 2 + [(37_000_000)] * 4 + \
             [(4096, 4096)] * 8 + [(12288, 4096)] * 2
    for sh in shapes:
        try:
            hog.append(torch.empty(sh, device="cuda", dtype=torch.uint8))
        except torch.OutOfMemoryError:
            break
    # 再制造碎片: 释放一半交错块
    for i in range(0, len(hog), 2):
        hog[i] = None
    torch.cuda.synchronize()
    print(f"占用后: alloc={torch.cuda.memory_allocated()/2**30:.2f}GB "
          f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GB")
    t1 = bench(lambda: vae.decode(lat, return_dict=False))
    print(f"占用环境 49块: {t1:.2f}s")

    torch.cuda.empty_cache()
    t2 = bench(lambda: vae.decode(lat, return_dict=False))
    print(f"empty_cache后: {t2:.2f}s")
print("SIM_DONE")
