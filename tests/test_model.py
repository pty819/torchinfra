# tests/test_model.py
import torch
from tests._ref import MODEL_DIR, PROMPT, hf_model, hf_tokenizer
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.kv_cache import KVCache


def main():
    tok = hf_tokenizer()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    assert ids.shape[1] >= 8

    # bf16 整模型 logits 对拍（公开 API，内部实现不依赖）
    ours = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    kv = KVCache(ours.cfg.num_hidden_layers, ours.max_seq,
                 ours.cfg.num_key_value_heads, ours.cfg.head_dim)
    with torch.no_grad():
        lo = ours(ids, kv=kv, start_pos=0, logits_to_keep="all").float()
    with torch.no_grad():
        lh = hf_model("bfloat16")(ids).logits.float()
    top1_match = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    assert top1_match >= 0.9, f"top-1 一致率过低: {top1_match}"
    assert (lo - lh).abs().max().item() < 1.0
    print(f"top-1 match {top1_match:.3f}, max|diff| {(lo - lh).abs().max().item():.3f}")

    # 分块等价（fp32 剥离 bf16 噪声）：prefill 前 16 + 逐 token 重放，末位 logits 一致
    del ours, kv, lo, lh  # 先释放 bf16 侧，避免 fp32 模型加载时 OOM
    torch.cuda.empty_cache()
    ours32 = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    kv2 = KVCache(ours32.cfg.num_hidden_layers, ours32.max_seq,
                  ours32.cfg.num_key_value_heads, ours32.cfg.head_dim,
                  dtype=torch.float32)
    n = 16
    with torch.no_grad():
        lo32 = ours32(ids, kv=kv2, start_pos=0, logits_to_keep="all").float()
        kv2 = KVCache(ours32.cfg.num_hidden_layers, ours32.max_seq,
                      ours32.cfg.num_key_value_heads, ours32.cfg.head_dim,
                      dtype=torch.float32)
        ours32(ids[:, :n], kv=kv2, start_pos=0, logits_to_keep="last")
        for i in range(n, ids.shape[1]):  # 重放 prompt 余下 token：喂 ids[i] 于位置 i
            last = ours32(ids[:, i:i + 1], kv=kv2, start_pos=i,
                          logits_to_keep="last")
        chunked_last = last.float()
    torch.testing.assert_close(chunked_last[:, -1], lo32[:, -1],
                               rtol=1e-3, atol=1e-3)
    print("model OK: 整段对拍 + 分块等价(fp32)都通过")


if __name__ == "__main__":
    main()
