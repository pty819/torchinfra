"""jina-reranker-v3.5：自研前向与官方 HF forward 的分数对拍。"""
import os

import pytest
import torch
from transformers import AutoModel

from nemotron_serving.jina_rerank import JinaReranker, _format_prompt

pytestmark = pytest.mark.gpu

PATH = os.path.expanduser("~/models/jina-reranker-v3.5")


def main():
    query = "What are the health benefits of green tea?"
    docs = [
        "Green tea contains catechins that may help reduce inflammation.",
        "El precio del cafe ha aumentado un 20% este ano.",
        "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险。",
        "Le the vert est riche en antioxydants.",
    ]
    ours = JinaReranker.from_pretrained(PATH, fp8=False)
    ids = ours.tok(_format_prompt(query, docs, None, ours.tok),
                   return_tensors="pt").input_ids.cuda()
    hf = AutoModel.from_pretrained(
        PATH, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    with torch.no_grad():
        got = ours.score_ids(ids).float()
        ref = hf(input_ids=ids).scores[0].float()
    err = (got - ref).abs().max().item()
    assert err < 0.01, err
    assert int(got.argmax()) == int(ref.argmax())

    # 单块 rerank 与官方 rerank 逐 index 对齐
    got_r = {r["index"]: r["relevance_score"] for r in ours.rerank(query, docs)}
    ref_r = {r["index"]: r["relevance_score"] for r in hf.rerank(query, docs)}
    err_r = max(abs(got_r[i] - ref_r[i]) for i in got_r)
    assert err_r < 0.01, err_r

    # 126 篇触发第二块（block_size=125）：加权平均 query 后再余弦
    many = [f"tea health note {i}" if i % 3 else f"unrelated coffee price {i}"
            for i in range(126)]
    got_m = {r["index"]: r["relevance_score"] for r in ours.rerank(query, many)}
    ref_m = {r["index"]: r["relevance_score"] for r in hf.rerank(query, many)}
    err_m = max(abs(got_m[i] - ref_m[i]) for i in got_m)
    assert err_m < 0.01, err_m

    # 长文档把单块推过 WINDOW=1024，覆盖滑窗 kernel 路径
    long_docs = [
        ("绿茶富含儿茶素，可以降低心脏病风险。" if i % 2 else
         "The coffee harvest this year was affected by drought. ") * 12
        for i in range(12)]
    got_l = {r["index"]: r["relevance_score"]
             for r in ours.rerank(query, long_docs)}
    ref_l = {r["index"]: r["relevance_score"]
             for r in hf.rerank(query, long_docs)}
    err_l = max(abs(got_l[i] - ref_l[i]) for i in got_l)
    assert err_l < 0.01, err_l
    print(f"jina rerank OK（单条 {err:.4f}，单块 {err_r:.4f}，两块 {err_m:.4f}，"
          f"滑窗长块 {err_l:.4f}）")


def test_jina_rerank_matches_hf():
    main()


if __name__ == "__main__":
    main()
