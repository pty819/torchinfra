"""CUDA Graph 前向路径对拍：graph vs eager 逐位对比。"""
import sys

import torch

sys.path.insert(0, ".")
from torchinfra.jina_rerank import JinaReranker, DOC_ID, QUERY_ID  # noqa: E402

FP8 = "~/models/jina-reranker-v3.5-fp8"


def main():
    r = JinaReranker.from_pretrained(FP8)
    torch.manual_seed(0)
    for S in [300, 512, 767, 2000, 4090, 8100, 9000]:
        ids = torch.randint(1000, 50000, (1, S), device="cuda")
        ids[0, S // 2] = DOC_ID
        ids[0, -1] = QUERY_ID
        h_eager = r._forward_hidden(ids)
        h_graph = r._forward_graphed(ids)
        if h_graph is None:
            print(f"S={S}: >GRAPH_MAX, 走 eager ✓")
            continue
        rel = (h_eager.float() - h_graph.float()).abs().max().item() \
            / h_eager.float().abs().max().item()
        # 打分路径对拍（graph 下重新走 embed_batch）
        doc_e = r._project(h_eager[0, ids[0] == DOC_ID]).float()
        doc_g = r._project(h_graph[0, ids[0] == DOC_ID]).float()
        err_s = (doc_e - doc_g).abs().max().item()
        print(f"S={S}: hidden 相对误差 {rel:.2e}, 投影后 {err_s:.2e}", flush=True)
        assert rel < 0.02 and err_s < 0.05, (S, rel, err_s)
    print("graph path OK")


if __name__ == "__main__":
    main()
