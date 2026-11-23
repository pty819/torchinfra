import os
"""CUDA Graph 前向路径对拍：graph vs eager。

两层门禁：
1. 桶精确 S（随机 id）：eager 与 graph 同 M、同 kernel、同算法，必须
   bit-exact——任何非 0 都意味着捕获/静态缓冲接线 bug。
2. 跨桶 S（真实文本）：tensorwise fp8 在 M 与桶填充 M 之间会命中不同
   GEMM 算法（~3e-3/层），hidden 空间放大到 ~1e-1 量级是已知的跨 M
   行为（README P11 实录），只约束投影归一化后的分数空间。
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from torchinfra.jina_rerank import JinaReranker, _format_prompt, DOC_ID  # noqa: E402

FP8 = os.path.expanduser("~/models/jina-reranker-v3.5-fp8")


def real_ids(r, n_docs, repeat):
    query = "What are the health benefits of green tea?"
    docs = [
        ("绿茶富含儿茶素，可以降低心脏病风险。" if i % 2 else
         "The coffee harvest this year was affected by drought. ") * repeat
        for i in range(n_docs)]
    return r.tok(_format_prompt(query, docs, None, r.tok),
                 return_tensors="pt").input_ids.cuda()


def main():
    r = JinaReranker.from_pretrained(FP8)

    # 门禁 1：桶精确 S，bit-exact
    torch.manual_seed(0)
    for S in [256, 1024, 4096, 8192]:
        ids = torch.randint(1000, 50000, (1, S), device="cuda")
        ids[0, S // 2] = DOC_ID
        ids[0, -1] = DOC_ID
        h_e = r._forward_hidden(ids)
        h_g = r._forward_graphed(ids)
        assert h_g is not None, S
        d = (h_e.float() - h_g.float()).abs().max().item()
        print(f"桶精确 S={S}: max diff {d:.2e}", flush=True)
        assert d == 0.0, (S, d)

    # 门禁 2：跨桶真实文本，约束分数空间
    for n_docs, repeat in [(2, 1), (8, 1), (12, 12)]:
        ids = real_ids(r, n_docs, repeat)
        h_e = r._forward_hidden(ids)
        h_g = r._forward_graphed(ids)
        if h_g is None:
            print(f"真实文本 {n_docs}x{repeat}: >GRAPH_MAX, 走 eager ✓")
            continue
        rel = (h_e.float() - h_g.float()).abs().max().item() \
            / h_e.float().abs().max().item()
        mask = ids[0] == DOC_ID
        p_e = F.normalize(r._project(h_e[0, mask]).float(), dim=-1)
        p_g = F.normalize(r._project(h_g[0, mask]).float(), dim=-1)
        err_s = (p_e - p_g).abs().max().item()
        print(f"真实文本 {n_docs}x{repeat}: S={ids.shape[1]} hidden rel "
              f"{rel:.2e}, 分数空间 {err_s:.2e}", flush=True)
        assert err_s < 0.02, (n_docs, repeat, err_s)
    print("graph path OK")


if __name__ == "__main__":
    main()
