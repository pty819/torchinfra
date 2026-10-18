"""HF tokenizer 薄封装：encode/decode/chat 模板/eos。"""
from pathlib import Path


class TokenizerBox:
    def __init__(self, model_dir: str | Path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        eos = {self.tok.eos_token_id}
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(model_dir)
            if gc.eos_token_id is not None:
                ids = gc.eos_token_id
                eos |= set(ids) if isinstance(ids, (list, tuple)) else {ids}
        except Exception:
            pass
        self.eos_ids = {int(e) for e in eos if e is not None}

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def apply_chat(self, messages: list[dict],
                   add_generation_prompt: bool = True) -> list[int]:
        try:
            out = self.tok.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt,
                tokenize=True, enable_thinking=False)
        except TypeError:  # 模板不支持 enable_thinking
            out = self.tok.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt,
                tokenize=True)
        # transformers 各版本返回类型不一：str / list[int] / BatchEncoding
        if isinstance(out, str):
            out = self.encode(out)
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if out and isinstance(out[0], list):  # batched
            out = out[0]
        return list(out)
