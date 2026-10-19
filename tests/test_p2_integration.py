# tests/test_p2_integration.py
"""P2 集成验收：backend="triton" 的引擎与 HF 对拍 + fp32 greedy 逐 token 一致。"""
import torch
from tests._ref import MODEL_DIR, PROMPT, hf_model, hf_tokenizer
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.kv_cache import KVCache
from torchinfra.generate import generate


def main():
    tok = hf_tokenizer()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    ours = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                            backend="triton")
    kv = KVCache(ours.cfg.num_hidden_layers, ours.max_seq,
                 ours.cfg.num_key_value_heads, ours.cfg.head_dim)
    with torch.no_grad():
        lo = ours(ids, kv=kv, start_pos=0, logits_to_keep="all").float()
    with torch.no_grad():
        lh = hf_model("bfloat16")(ids).logits.float()
    top1 = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    assert top1 >= 0.9, top1
    print(f"triton 引擎 logits 对拍 top-1 = {top1:.3f}")

    # 终局：fp32 + triton 后端，greedy 与 HF 逐 token 一致
    del ours, kv, lo, lh
    torch.cuda.empty_cache()
    ours32 = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                              backend="triton")
    from torchinfra.tokenizer import TokenizerBox
    tb = TokenizerBox(MODEL_DIR)
    pids = tb.encode(PROMPT)
    ours_ids = [t for t, _ in generate(ours32, tb, pids, max_new_tokens=16)]
    with torch.no_grad():
        out = hf_model("float32").generate(
            torch.tensor([pids], device="cuda"), max_new_tokens=16,
            do_sample=False, pad_token_id=tb.tok.eos_token_id)
    assert ours_ids == out[0, len(pids):].tolist(), \
        f"ours={ours_ids}\nhf  ={out[0, len(pids):].tolist()}"
    print("triton 引擎 fp32 greedy 16 token 与 HF 逐 token 一致")


if __name__ == "__main__":
    main()
