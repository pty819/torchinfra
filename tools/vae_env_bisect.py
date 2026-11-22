# worker 环境税二分：逐级 import/编译环境，176² 小图 decode 计时
import os, sys, time, json
import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "base"

if MODE in ("dynamo", "full"):
    import torch._dynamo
    f = torch.compile(lambda a: a * 2 + 1)
    f(torch.randn(64, 64, device="cuda"))
    torch.cuda.synchronize()
if MODE == "full":
    sys.path.insert(0, "~/torchinfra")
    import triton
    import flydsl_kernels
    from flydsl_kernels.fp8_wrapper import quantize_tensorwise
    q = torch.randn(4, 4, 32, 128, device="cuda", dtype=torch.bfloat16)
    quantize_tensorwise(q)

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

H = W = 11
g = torch.Generator(device='cuda').manual_seed(7)
lat = torch.randn(1, 64, 1, H, W, device='cuda', dtype=torch.bfloat16, generator=g) * 3
mean = torch.tensor(cfgd["latents_mean"], device="cuda", dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
std = torch.tensor(cfgd["latents_std"], device="cuda", dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
lat = lat * std + mean

def bench(fn, n=10, warm=3):
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
    vae.disable_tiling()
    t = bench(lambda: vae.decode(lat, return_dict=False))
    print(f"[{MODE}] 176² 无tiling: {t*1000:.1f}ms")
    vae.enable_tiling(tile_sample_min_height=64, tile_sample_min_width=64,
                      tile_sample_stride_height=48, tile_sample_stride_width=48)
    t2 = bench(lambda: vae.decode(lat, return_dict=False))
    print(f"[{MODE}] 176² 16块tiling: {t2*1000:.1f}ms")
print("BISECT_DONE")
