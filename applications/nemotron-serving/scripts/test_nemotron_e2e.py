"""E2E 对拍：torchinfra fp8 移植 vs CPU bf16 transformers 参考（8 对）。

覆盖三条执行路径：
1. B=1 小 S → CUDA Graph（自研双向 kernel + 桶位右 padding）
2. B=1 大 S → eager SDPA flash（无 padding）
3. B>1 右 padding batch → 自研双向 kernel（vlen 掩码）
"""
import json, sys
import torch

sys.path.insert(0, ".")
from nemotron_serving.nemotron_rerank import NemotronReranker

import os
MODEL = os.environ.get("NEMO_MODEL", os.path.expanduser("~/models/Nemotron-3.5-Rerank-8B-FP8"))

ref = json.load(open("/tmp/nemotron_ref.json"))
print(f"参考 {len(ref)} 对")
rr = NemotronReranker.from_pretrained(MODEL)
free, _ = torch.cuda.mem_get_info()
print(f"加载后 VRAM 剩余 {free/2**30:.2f} GiB")

# ---- 路径 1/2：逐对 B=1（小 S 自动走图，大 S eager flash）----
pairs = []
for r in ref:
    ids = torch.tensor([r["ids"]], device="cuda")
    ours = float(rr.score_ids(ids))
    pairs.append((r["logit"], ours, ids.shape[1]))

print("\n对拍（参考 bf16 vs 移植 fp8）:")
print(f"{'S':>5} {'参考':>10} {'移植':>10} {'Δ':>8}")
for ref_v, our_v, S in pairs:
    print(f"{S:>5} {ref_v:>10.4f} {our_v:>10.4f} {our_v-ref_v:>+8.4f}")

deltas = [o - r for r, o, _ in pairs]
max_abs = max(abs(d) for d in deltas)
# 排序保序性：两组 logit 的排序应一致（相关性 = 1）
import statistics
ref_order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
our_order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
order_ok = ref_order == our_order
print(f"\nmax|Δ|={max_abs:.4f}  排序一致={order_ok}")
assert max_abs < 0.30, "对拍偏差过大"
assert order_ok, "打分排序被量化/移植破坏"

# ---- 路径 3：右 padding batch vs 单条 ----
a, b = ref[0], ref[1]     # S=23 / S=25 量级
Sa, Sb_ = len(a["ids"]), len(b["ids"])
S = max(Sa, Sb_)
ids = torch.zeros(2, S, dtype=torch.long, device="cuda")
mask = torch.zeros(2, S, dtype=torch.long, device="cuda")
ids[0, :Sa] = torch.tensor(a["ids"], device="cuda"); mask[0, :Sa] = 1
ids[1, :Sb_] = torch.tensor(b["ids"], device="cuda"); mask[1, :Sb_] = 1
batched = rr.score_ids(ids, mask).tolist()
singles = [float(rr.score_ids(torch.tensor([a["ids"]], device="cuda"))),
           float(rr.score_ids(torch.tensor([b["ids"]], device="cuda")))]
print(f"\npadding batch: {batched} vs 单条 {singles}  "
      f"Δ={[abs(x-y) for x, y in zip(batched, singles)]}")
# tensorwise fp8 的 scale 是整批一个：批组成变了 amax 就变，另一个文档的
# 量化步长跟着变 → Δ 不是 0。逐条各自离 bf16 参考 <0.3、彼此排序不变
# 才是硬要求（见上），batch 内部一致性放宽到 0.05。
d_ba = max(abs(x - y) for x, y in zip(batched, singles))
assert d_ba < 0.05, f"右 padding batch 与单条偏差 {d_ba:.4f} 超预期"
assert (batched[0] > batched[1]) == (singles[0] > singles[1]), "batch 内排序翻转"

# ---- 等长 batch（B>1 无 mask flash 路径）----
# 拆 scale 后 k/v 是 N=1024 的瘦 GEMM，hipBLASLt 对这种形状选 split-k
# 原子归约 kernel：批内两行相同输入的原子加法顺序不同 → ~1e-3 级 fp
# 抖动（融合 N=6144 时代是逐位一致的）。门禁从 bit-exact 放宽到 2e-3
# （对拍阈值 0.3 的百分之一量级），排序不受影响。
ids2 = torch.tensor([a["ids"], a["ids"]], device="cuda")
eq = rr.score_ids(ids2).tolist()
print(f"等长 batch: {eq}（Δ={abs(eq[0]-eq[1]):.2e}，split-k 原子归约抖动）")
assert abs(eq[0] - eq[1]) < 2e-3

print("\nE2E 对拍全部通过")
