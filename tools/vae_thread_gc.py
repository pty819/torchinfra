# 候选①: decode 在子线程跑; 候选②: 巨型 Python 堆 (GC) 拖慢 tiling 循环
import time, json, threading
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

H = W = 84
g = torch.Generator(device='cuda').manual_seed(7)
lat = torch.randn(1, 64, 1, H, W, device='cuda', dtype=torch.bfloat16, generator=g) * 3
mean = torch.tensor(cfgd["latents_mean"], device="cuda", dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
std = torch.tensor(cfgd["latents_std"], device="cuda", dtype=torch.bfloat16).view(1, 64, 1, 1, 1)
lat = lat * std + mean

def bench(n=3, warm=1):
    with torch.no_grad():
        for _ in range(warm):
            vae.decode(lat, return_dict=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            vae.decode(lat, return_dict=False)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

with torch.no_grad():
    vae.enable_tiling()
    print(f"主线程 49块: {bench():.2f}s")

    res = {}
    def run_thread():
        res['t'] = bench()
    th = threading.Thread(target=run_thread)
    th.start(); th.join()
    print(f"子线程 49块: {res['t']:.2f}s")

    # 候选②: 构造巨型 Python 堆 (模拟 dynamo 编译产物 + 长寿对象)
    import gc
    bigheap = []
    for i in range(3_000_000):
        bigheap.append((f"graph_module_{i}", [i, i + 1, {"guard": i}]))
        if len(bigheap) % 500000 == 0:
            print(f"  堆对象 {len(bigheap)}...")
    print(f"GC 对象数: {len(gc.get_objects())}")
    t = bench()
    print(f"巨型堆下 49块: {t:.2f}s")
    # 手动触发一次 full GC 计时
    t0 = time.perf_counter(); gc.collect(); print(f"  gc.collect() 耗时 {time.perf_counter()-t0:.2f}s")
    del bigheap
print("THREAD_GC_DONE")
