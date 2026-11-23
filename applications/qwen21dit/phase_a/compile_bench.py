import torch, time
from loader import load_fp8
from sample import sample
from sigmas import flux_sigmas
print("loading...", flush=True)
model = load_fp8(device="cuda", mode="fp8")
model = torch.compile(model)
ctx = torch.randn(1, 64, 4096, device="cuda").to(torch.bfloat16) * 0.5
neg = torch.zeros_like(ctx)
sig = flux_sigmas(20, 7056).tolist()
t0 = time.time()
print("warmup(cfg4, 触发编译)...", flush=True)
x = sample(model, ctx, neg, (16, 168, 168), sig, cfg=4.0, seed=7, device="cuda")
torch.cuda.synchronize()
print(f"编译+首跑: {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
x = sample(model, ctx, neg, (16, 168, 168), sig, cfg=4.0, seed=7, device="cuda")
torch.cuda.synchronize()
print(f"Gate口径 compiled: {time.time()-t0:.1f}s | finite={torch.isfinite(x).all().item()}", flush=True)
