"""contiguous KV cache：单请求、每层预分配 [max_seq, n_kv, head_dim]。"""
import torch


class KVCache:
    def __init__(self, n_layers: int, max_seq: int, n_kv_heads: int,
                 head_dim: int, dtype=torch.bfloat16, device: str = "cuda"):
        self.k = torch.zeros(n_layers, max_seq, n_kv_heads, head_dim,
                             dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.max_seq = max_seq

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
               start: int) -> None:
        """k/v 形状 [1, n_kv, s, head_dim]，写入 [start, start+s)。"""
        s = k.shape[2]
        self.k[layer_idx, start:start + s] = k[0].permute(1, 0, 2)
        self.v[layer_idx, start:start + s] = v[0].permute(1, 0, 2)

    def read(self, layer_idx: int, upto: int):
        """返回 [1, n_kv, upto, head_dim] 视图。"""
        k = self.k[layer_idx, :upto].unsqueeze(0).permute(0, 2, 1, 3)
        v = self.v[layer_idx, :upto].unsqueeze(0).permute(0, 2, 1, 3)
        return k, v
