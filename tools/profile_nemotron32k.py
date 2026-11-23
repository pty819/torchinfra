"""Nemotron 32K 单序列 e2e 热点剖析：kernel 级时间按功能桶聚合。

用法：PYTHONPATH=. python tools/profile_nemotron32k.py [S]
"""
import os
import re
import statistics
import sys
import time

sys.path.insert(0, ".")
import torch

from torchinfra.nemotron_rerank import NemotronReranker

MODEL = os.environ.get("NEMO_MODEL", os.path.expanduser("~/models/Nemotron-3.5-Rerank-8B-FP8"))
S = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
N_LAYERS = 34

SKIP = re.compile(r"^(hip|cuda|aten::|Profiler|autograd|Memcpy|Memset)|__pyx_nespace", re.I)
BUCKETS = [
    ("fp8注意力(flydsl)", lambda n: "_kernel_legacy" in n or "_kernel_pipe" in n or "flash_attn_fp8" in n),
    ("fp8 GEMM(rocblas)", lambda n: re.search(r"gemm|blas|scaled_mm|Cijk_", n)),
    ("rope+量化(triton)", lambda n: "rope" in n),
    ("norm(rmsnorm)", lambda n: "rmsnorm" in n),
    ("激活(silu)", lambda n: "silu" in n),
    ("amax/归约", lambda n: re.search(r"amax|reduce|AbsMax|Max", n)),
    ("拷贝/填充", lambda n: re.search(r"copy|cat|fill|flip|index|slice|elementwise|vectorized|triu", n, re.I)),
]


def bucket_of(name: str) -> str:
    for label, pred in BUCKETS:
        if pred(name):
            return label
    return "其他"


def main():
    rr = NemotronReranker.from_pretrained(MODEL)
    torch.manual_seed(0)
    ids = torch.randint(3, 130000, (1, S), device="cuda")

    for _ in range(2):
        rr.score_ids(ids)
    torch.cuda.synchronize()

    # 墙钟（不含 profiler 开销）
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        rr.score_ids(ids)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    wall = statistics.median(ts)

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            rr.score_ids(ids)
        torch.cuda.synchronize()

    agg = {}
    gpu_total = 0.0
    n_fwd = 3
    gemm_shapes = {}
    for ev in prof.key_averages(group_by_input_shape=True):
        if "_scaled_mm" in ev.key and ev.self_device_time_total > 0:
            shapes = str(ev.input_shapes or "")
            if shapes:
                gemm_shapes[shapes] = gemm_shapes.get(shapes, 0.0) + ev.self_device_time_total
    for ev in prof.key_averages():
        if ev.self_device_time_total <= 0:
            continue
        if SKIP.search(ev.key):
            continue
        gpu_total += ev.self_device_time_total
        agg.setdefault(bucket_of(ev.key), [0.0, 0, {}])
        agg[bucket_of(ev.key)][0] += ev.self_device_time_total
        agg[bucket_of(ev.key)][1] += ev.count
        inner = agg[bucket_of(ev.key)][2]
        inner[ev.key] = inner.get(ev.key, 0.0) + ev.self_device_time_total

    gpu_total /= n_fwd * 1e3
    print(f"\nS={S}  e2e 墙钟 {wall:.0f}ms/条 | GPU busy {gpu_total:.0f}ms "
          f"| launch/同步间隙 {wall - gpu_total:.0f}ms ({(wall-gpu_total)/wall*100:.0f}%)\n")
    print(f"{'功能桶':<22}{'ms/条':>9}{'%GPU':>7}{'次数':>8}   说明")
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    for label, (t, cnt, inner) in rows:
        tms = t / n_fwd / 1e3
        note = ""
        if "flydsl" in label:
            note = f"≈{tms/N_LAYERS:.1f}ms/层"
        elif "GEMM" in label:
            note = f"≈{tms/N_LAYERS:.2f}ms/层（34 层×7 个投影）"
        elif "rope" in label:
            note = f"≈{tms/N_LAYERS:.2f}ms/层（amax+量化两遍）"
        print(f"{label:<22}{tms:>9.1f}{tms/gpu_total*100:>6.1f}%{cnt//n_fwd:>8}   {note}")

    print("\n-- _scaled_mm 按输入形状 --")
    NAMEMAP = [("4096, 6144", "qkv"), ("4096, 4096", "o_proj/down?"), ("8192, 16384", "gate_up"), ("4096, 16384", "gate_up"), ("8192, 4096", "down")]
    for shp, tms in sorted(gemm_shapes.items(), key=lambda kv: -kv[1])[:8]:
        tag = next((nm for pat, nm in NAMEMAP if pat in shp), "")
        print(f"  {tms/n_fwd/1e3:8.1f}ms  {tag:<12} {shp[:70]}")

    print("\n-- 单 kernel Top12 --")
    top = sorted(((t / n_fwd / 1e3, k) for k, t in
                  ((k, v) for _, (_, _, m) in rows for k, v in m.items())),
                 reverse=True)[:12]
    for t, k in top:
        print(f"  {t:8.1f}ms  {k[:88]}")

    # 每层注意力 FLOPs 口径
    fl = 4 * 32 * S * S * 128 * N_LAYERS
    attn = agg.get("fp8注意力(flydsl)", [0])[0] / n_fwd / 1e3
    if attn > 0:
        print(f"\n注意力有效吞吐: {fl/(attn*1e-3)/1e12:.1f} TF（含 34 层）")


if __name__ == "__main__":
    main()
