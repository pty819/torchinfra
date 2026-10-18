# tests/test_p1_acceptance.py
"""P1 终局验收：与 transformers 同 prompt 同参数生成逐 token 一致。"""
import torch
from tests._ref import MODEL_DIR, PROMPT, hf_model, hf_tokenizer
from torchinfra.model import Qwen3ForCausalLM
from torchinfra.tokenizer import TokenizerBox
from torchinfra.generate import generate


def main():
    N = 32
    tb = TokenizerBox(MODEL_DIR)
    ids = tb.encode(PROMPT)

    ours = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    ours_ids = [t for t, _ in generate(ours, tb, ids, max_new_tokens=N)]

    hf = hf_model("float32")
    with torch.no_grad():
        out = hf.generate(torch.tensor([ids], device="cuda"),
                          max_new_tokens=N, do_sample=False,
                          pad_token_id=tb.tok.eos_token_id)
    hf_ids = out[0, len(ids):].tolist()

    assert ours_ids == hf_ids, \
        f"不一致:\nours={ours_ids}\nhf  ={hf_ids}"
    print(f"acceptance OK: {N} tokens 逐 token 一致")
    print("生成内容:", tb.decode(ids + ours_ids)[:200])


if __name__ == "__main__":
    main()
