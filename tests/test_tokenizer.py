# tests/test_tokenizer.py
from pathlib import Path
from torchinfra.tokenizer import TokenizerBox

MODEL_DIR = Path.home() / "models" / "Qwen3-0.6B"


def main():
    tb = TokenizerBox(MODEL_DIR)
    s = "你好，world！Paged attention 是什么？"
    ids = tb.encode(s)
    assert len(ids) > 5
    assert tb.decode(ids) == s
    chat_ids = tb.apply_chat([{"role": "user", "content": "你好"}])
    # 结构断言用 raw decode（decode 默认剥 special tokens，对生成输出是对的）
    raw = tb.tok.decode(chat_ids, skip_special_tokens=False)
    assert "<|im_start|>user" in raw and "<|im_end|>" in raw
    assert "<think>" in raw and "</think>" in raw  # enable_thinking=False 的空 think 块
    assert raw.endswith("<|im_start|>assistant")
    assert isinstance(tb.eos_ids, set) and len(tb.eos_ids) >= 1
    assert tb.eos_ids == {151643, 151645}  # <|endoftext|> + <|im_end|>
    print("tokenizer OK, eos_ids =", tb.eos_ids)


if __name__ == "__main__":
    main()
