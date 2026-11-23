import os
# torch.compile(VAE decoder) 验证: 176² 时间 + 数值
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
    # eager 基线
    vae.disable_tiling()
    t_eager = bench(lambda: vae.decode(lat, return_dict=False))
    o_eager = vae.decode(lat, return_dict=False)[0].float()
    print(f"eager 176²: {t_eager*1000:.1f}ms")

    # compile decoder (整 decode 含 tiling 循环不走 compile, 只编译单次 forward)
    import torch._dynamo as _dyn
    _dyn.config.suppress_errors = True
    dec_c = torch.compile(vae.decoder, dynamic=False)
    t0c = time.perf_counter()
    with torch.no_grad():
        o_c0 = dec_c(lat.clone())
    torch.cuda.synchronize()
    print(f"compile 冷启动: {time.perf_counter()-t0c:.1f}s")
    t_comp = bench(lambda: dec_c(lat.clone()))
    o_comp = dec_c(lat.clone()).float()
    print(f"compiled 176²: {t_comp*1000:.1f}ms  ({t_eager/t_comp:.2f}x)")
    print(f"Δ(eager vs compiled): {(o_eager - o_comp).abs().max().item():.5f}")
print("COMPILE_TEST_DONE")
