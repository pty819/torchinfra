"""rerank 前向的 kernel 级剖析。用法：python profile_rerank.py <S>"""
import sys

import torch

sys.path.insert(0, ".")
from torchinfra.jina_rerank import JinaReranker  # noqa: E402

FP8 = "~/models/jina-reranker-v3.5-fp8"


def main():
    S = int(sys.argv[1])
    r = JinaReranker.from_pretrained(FP8)
    torch.manual_seed(0)
    ids = torch.randint(1000, 50000, (1, S), device="cuda")
    for _ in range(2):  # warmup
        r._forward_hidden(ids)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            r._forward_hidden(ids)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18,
                                    max_name_column_width=60))


if __name__ == "__main__":
    main()
