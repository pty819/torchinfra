"""测试共享：HF 参考模型懒加载（进程内只加载一次）+ 固定 prompt。"""
from functools import lru_cache
from pathlib import Path
import torch

MODEL_DIR = Path.home() / "models" / "Qwen3-0.6B"
PROMPT = ("请用中文详细解释一下大模型推理系统里的paged attention是什么，"
          "它解决了 KV cache 管理中的什么问题，以及它和普通 attention 的区别。")


@lru_cache(maxsize=2)
def hf_model(dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dt)
    return m.cuda().eval()


@lru_cache(maxsize=1)
def hf_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)
