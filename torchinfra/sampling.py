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


def sample_batch(logits: torch.Tensor, temperature=0.0, top_k: int = 0,
                 top_p=0.0, generator: torch.Generator | None = None
                 ) -> list[int]:
    """整批采样：logits [N, V]，每请求参数可为标量或 [N] 序列。

    全程 GPU 批处理，最后单次 tolist 同步——替代 N 次 sample_token 的
    Python 循环（ scheduler decode 每步省 ~3.5ms）。
    temperature/top_p 支持逐行；greedy 行（temperature<=0）直接 argmax。
    top_k 仅支持整批同值。
    """
    logits = logits.float()
    n = logits.shape[0]

    def _vec(v, default):
        if v is None:
            v = default
        if torch.is_tensor(v):
            t = v.float().reshape(n, 1).to(logits.device)
        elif isinstance(v, (list, tuple)):
            t = torch.tensor(v, dtype=torch.float32,
                             device=logits.device).reshape(n, 1)
        else:
            t = torch.full((n, 1), float(v), device=logits.device)
        return t

    temp = _vec(temperature, 0.0)
    tp = _vec(top_p, 0.0)
    greedy = temp <= 0                                  # [N,1]

    z = logits / torch.where(greedy, torch.ones_like(temp), temp)
    if 0 < top_k < logits.shape[-1]:
        kth = torch.topk(z, top_k, dim=-1).values[..., -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    if bool((tp > 0).any()):
        sz, idx = torch.sort(z, dim=-1, descending=True)
        pr = torch.softmax(sz, dim=-1)
        keep = (pr.cumsum(-1) - pr) <= tp               # [N,V]，HF 风格截断
        z = z.scatter(1, idx, torch.where(keep, sz, float("-inf")))

    probs = torch.softmax(z, dim=-1)
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    out = torch.where(greedy.squeeze(1), logits.argmax(-1), sampled)
    return out.tolist()
