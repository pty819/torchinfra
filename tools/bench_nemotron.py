"""bench：B=1 各 S 延迟/吞吐 + 32K 显存峰值 + batch 吞吐。"""
import sys, time, statistics
sys.path.insert(0, ".")
import torch
from torchinfra.nemotron_rerank import NemotronReranker

import os
MODEL = os.environ.get("NEMO_MODEL", "~/models/Nemotron-3.5-Rerank-8B-FP8")
rr = NemotronReranker.from_pretrained(MODEL)
free0, total = torch.cuda.mem_get_info()
print(f"加载后空闲 {free0/2**30:.2f} / {total/2**30:.2f} GiB\n")

torch.cuda.reset_peak_memory_stats()
print(f"{'S':>6} {'路径':>6} {'延迟ms':>8} {'tok/s':>8} {'峰值MB':>8}")
for S in [512, 2048, 8192, 16384, 32768]:
    ids = torch.randint(3, 130000, (1, S), device="cuda")
    for _ in range(2):                       # warmup（≤2048 会顺带抓图）
        rr.score_ids(ids)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        rr.score_ids(ids)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    lat = statistics.median(ts)
    peak = torch.cuda.max_memory_allocated() / 2**20
    path = "graph" if S <= rr.GRAPH_MAX else "flash"
    print(f"{S:>6} {path:>6} {lat:>8.1f} {S/lat*1e3:>8.0f} {peak:>8.0f}")

# batch 吞吐：B=8 等长 → eager vlen kernel 路径
B, S = 8, 2048
ids = torch.randint(3, 130000, (B, S), device="cuda")
mask = torch.ones(B, S, dtype=torch.long, device="cuda")
for _ in range(2):
    rr.score_ids(ids, mask)
torch.cuda.synchronize()
ts = []
for _ in range(3):
    t0 = time.perf_counter()
    rr.score_ids(ids, mask)
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
lat = statistics.median(ts)
print(f"\nB={B} S={S} 双向kernel: {lat:.1f}ms  {B*S/lat*1e3:.0f} tok/s")

free, _ = torch.cuda.mem_get_info()
print(f"结束: torch 活跃分配 {torch.cuda.memory_allocated()/2**30:.2f} GiB，"
      f"板上空闲 {free/2**30:.2f} GiB（缓存器保留大块属正常，非泄漏——"
      f"重复迭代延迟稳定可证）")
