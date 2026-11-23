import os
"""reranker 前向的 torch.compile 实验：延迟 + 数值对拍（fp8 路径）。

对拍是硬门（P8 曾有 _fused_kernel 深层链翻转 bug 的前科）：
compile 输出与 eager 的 hidden 相对误差必须 < 1e-2（真实量级下）。
"""
import sys
import time

import torch

sys.path.insert(0, ".")
from nemotron_serving.jina_rerank import JinaReranker  # noqa: E402

FP8 = os.path.expanduser("~/models/jina-reranker-v3.5-fp8")


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    r = JinaReranker.from_pretrained(FP8)
    torch.manual_seed(0)
    # 半真实输入：用中等幅度的随机 id（避免全随机 id 的病态激活掩盖数值问题）
    for S in [767, 8782]:
        ids = torch.randint(1000, 50000, (1, S), device="cuda")
        he = r._forward_hidden(ids)
        t_eager = bench(lambda: r._forward_hidden(ids), n=10)
        print(f"S={S} eager: {t_eager:.2f} ms", flush=True)

        for mode in [None, "reduce-overhead"]:
            tag = mode or "default"
            try:
                fn = torch.compile(r._forward_hidden, dynamic=False, mode=mode)
                t0 = time.perf_counter()
                hc = fn(ids)  # 触发编译
                torch.cuda.synchronize()
                ctime = time.perf_counter() - t0
                rel = (he.float() - hc.float()).abs().max().item() \
                    / he.float().abs().max().item()
                t_c = bench(lambda: fn(ids), n=10)
                print(f"S={S} compile[{tag}]: 编译 {ctime:.0f}s, "
                      f"{t_c:.2f} ms ({t_eager/t_c:.2f}x), "
                      f"相对误差 {rel:.2e}", flush=True)
                assert rel < 1e-2, (S, tag, rel)
            except Exception as e:
                print(f"S={S} compile[{tag}]: FAIL — "
                      f"{type(e).__name__}: {str(e)[:160]}", flush=True)
                # 编译失败不拖垮后续用例
                torch.cuda.synchronize()


if __name__ == "__main__":
    main()
