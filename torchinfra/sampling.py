"""采样：greedy / temperature / top-k / top-p（HF 风格截断）。"""
import torch


def sample_token(logits: torch.Tensor, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 0.0,
                 generator: torch.Generator | None = None) -> int:
    logits = logits.float()
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        keep = (probs.cumsum(-1) - probs) <= top_p  # 首个越界 token 仍保留
        logits = torch.full_like(logits, float("-inf"))
        logits[idx[keep]] = sorted_logits[keep]
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
