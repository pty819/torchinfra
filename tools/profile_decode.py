"""decode 热循环剖析：把每步 wall 时长拆成 纯 GPU 时长 / CPU 串行间隙。

三种口径：
  A 现状路径     —— scheduler.step() 原样（含 staging 重建 + 采样同步）
  B 零同步模拟   —— staging 只做一次，循环内纯 GPU enqueue（lens.add_ /
                   replay / argmax / tokens 回填 / pinned 异步拷贝），无任何同步
  C kernel 分布  —— torch.profiler 采 A 路径若干步，列出热点 kernel

用法：PYTHONPATH=. python tools/profile_decode.py [batch]
"""
import sys
import time
import torch

from torchinfra.model import Qwen3ForCausalLM
from torchinfra.tokenizer import TokenizerBox
from torchinfra.scheduler import Scheduler

MODEL_DIR = "~/models/Qwen3-0.6B"
N_STEPS = 100


def build(batch: int) -> tuple[Scheduler, int]:
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR)
    tok = TokenizerBox(MODEL_DIR)
    sched = Scheduler(model, tok, num_blocks=max(384, batch * 24),
                      max_seqs=batch, graph_mode="piecewise",
                      buckets=tuple([batch]))
    sched.warmup()
    for i in range(batch):
        sched.submit(tok.encode(f"第{i}个请求：请简单介绍一下你自己。"),
                     max_new_tokens=256)
    # 推进到全部进入 decode 稳态（prefill 逐个完成）
    while any(r.state == "waiting" for r in sched._waiting) or sched._waiting:
        sched.step()
    return sched, batch


def phase_a(sched: Scheduler, batch: int) -> float:
    """现状：原样 step()，含全部 staging 与采样同步。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        sched.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_STEPS * 1000


def phase_b(sched: Scheduler, batch: int) -> tuple[float, float]:
    """零同步模拟：staging 一次，之后纯 enqueue。"""
    run = sched._runners[batch]
    reqs = sched._running
    rows = [(r.rid, r.kv_len) for r in reqs]
    n = len(rows)

    def _replay() -> torch.Tensor:
        if hasattr(run, "graph"):                     # manual 整图
            run.graph.replay()
            return run.out[:n]
        return run._fn()[:n]                          # piecewise

    with torch.no_grad():
        run.run([r.output_ids[-1] for r in reqs], rows)   # staging + 预热回放
        torch.cuda.synchronize()

        pin = torch.zeros(batch, dtype=torch.long, pin_memory=True)
        ev = torch.cuda.Event()

        gpu0 = torch.cuda.Event(enable_timing=True)
        gpu1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        gpu0.record()
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            # 稳态每步：lens 前进 → replay → 采样（greedy=argmax）→ 回填 tokens
            run.lens.add_(1)
            torch.add(run.lens, 1, out=run.lens_p1)
            ids = _replay().argmax(-1)
            run.tokens[:n, 0].copy_(ids)
            pin[:n].copy_(ids, non_blocking=True)       # lag-1 完成检测
            ev.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / N_STEPS * 1000
        gpu1.record()
        torch.cuda.synchronize()
        gpu = gpu0.elapsed_time(gpu1) / N_STEPS
    return wall, gpu


def phase_c(sched: Scheduler, batch: int) -> None:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            sched.step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sched, batch = build(batch)
    a = phase_a(sched, batch)
    b_wall, b_gpu = phase_b(sched, batch)
    print(f"\nB={batch}  每步（ms）：")
    print(f"  A 现状（含同步与 staging）     {a:6.2f}")
    print(f"  B 零同步模拟 wall              {b_wall:6.2f}")
    print(f"  B 零同步模拟 GPU span          {b_gpu:6.2f}")
    print(f"  → overlap 可省                 {a - b_wall:6.2f} ms/步"
          f"（{(a / b_wall - 1) * 100:.0f}% 提速空间）")
    print(f"  → CPU 间隙（wall-GPU）         {b_wall - b_gpu:6.2f}"
          f"（>0 说明 kernel 间仍有空泡）")
    print("\n---- kernel 分布（A 路径 20 步）----")
    phase_c(sched, batch)


if __name__ == "__main__":
    main()
