"""稳态 decode 循环分段计时：定位未被 GPU 遮蔽的 CPU 段。"""
import sys
import time
import torch

from tests._ref import MODEL_DIR
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.scheduler import Scheduler
from torchinfra.tokenizer import TokenizerBox


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    MODE = sys.argv[2] if len(sys.argv) > 2 else "piecewise"
    torch.manual_seed(0)
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR)
    tok = TokenizerBox(MODEL_DIR)
    sched = Scheduler(model, tok, num_blocks=max(64, B * 24), max_seqs=B,
                      graph_mode=MODE, buckets=(B,))
    sched.warmup()
    for i in range(B):
        sched.submit(tok.encode("写一段关于秋天的文字。"), max_new_tokens=300)
    while sched._waiting:
        sched.step()

    n = 200
    t_c = [0.0]; t_r = [0.0]; t_s = [0.0]; t_f = [0.0]; t_p = [0.0]
    torch.cuda.synchronize()
    T0 = time.perf_counter()
    with torch.no_grad():
      for _ in range(n):
        t0 = time.perf_counter(); sched._try_consume()
        t_c[0] += time.perf_counter() - t0
        st = sched._steady
        if st is None:
            sched._start_steady()
            continue
        run = st["runner"]
        run.advance()
        t0 = time.perf_counter()
        logits = run.replay()[:st["n"]]
        t_r[0] += time.perf_counter() - t0
        t0 = time.perf_counter()
        ids = st["sampler"](logits)
        t_s[0] += time.perf_counter() - t0
        t0 = time.perf_counter()
        run.feed(ids)
        t_f[0] += time.perf_counter() - t0
        t0 = time.perf_counter()
        p = st["par"]
        st["pin"][p][:st["n"]].copy_(ids, non_blocking=True)
        st["ev"][p].record()
        st["pend"][p] = {"pin": st["pin"][p], "ev": st["ev"][p],
                         "reqs": st["reqs"], "n": st["n"]}
        st["par"] = p ^ 1
        for r in st["reqs"]:
            r.kv_len += 1
        t_p[0] += time.perf_counter() - t0
    torch.cuda.synchronize()
    wall = (time.perf_counter() - T0) / n * 1e3
    f = lambda x: x / n * 1e3
    print(f"B={B} wall {wall:.2f}ms/步 | try_consume {f(t_c[0]):.3f} | "
          f"replay_enq {f(t_r[0]):.3f} | sampler {f(t_s[0]):.3f} | "
          f"feed {f(t_f[0]):.3f} | pin+record {f(t_p[0]):.3f}")


if __name__ == "__main__":
    main()
