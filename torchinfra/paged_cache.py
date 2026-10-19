"""分页 KV cache：物理块池 + block_table 两级映射（vLLM B_Loc 模式）。

存储布局 k/v: [n_layers, num_blocks, n_kv_heads, block_size, head_dim]。
free list 用 python list（LIFO），分配走 table 查表——教学版不追求分配性能。
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class CacheView:
    """传给 Triton kernel 的只读视图（T4 定义 kernel 时消费）。"""
    k: torch.Tensor            # [n_layers, num_blocks, n_kv, bs, hd]
    v: torch.Tensor
    block_table: torch.Tensor  # [max_seqs, max_blocks_per_seq] int32, -1 填充
    kv_lens: torch.Tensor      # [max_seqs] int32
    block_size: int

    def slice_batch(self, ids: list[int]) -> "CacheView":
        return CacheView(self.k, self.v, self.block_table[ids],
                         self.kv_lens[ids], self.block_size)


class PagedKVCache:
    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 num_blocks: int, block_size: int,
                 dtype=torch.bfloat16, device: str = "cuda",
                 max_seqs: int = 8):
        self.n_layers = n_layers
        self.n_kv = n_kv_heads
        self.hd = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.max_blocks_per_seq = num_blocks  # 上界（共享池约束实际总量）
        self.k = torch.zeros(n_layers, num_blocks, n_kv_heads, block_size,
                             head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.block_table = torch.full((max_seqs, self.max_blocks_per_seq),
                                      -1, dtype=torch.int32, device=device)
        self.kv_lens = torch.zeros(max_seqs, dtype=torch.int32, device=device)
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # LIFO

    # ---- 分配 / 释放 -------------------------------------------------
    def alloc(self, req_id: int, n_tokens: int) -> int:
        need = math.ceil(n_tokens / self.block_size)
        assert len(self._free) >= need, "块池耗尽"
        blocks = [self._free.pop() for _ in range(need)]
        self.block_table[req_id, :need] = torch.tensor(blocks,
                                                       dtype=torch.int32,
                                                       device=self.k.device)
        return need

    def extend(self, req_id: int, n_new_tokens: int) -> None:
        cur = int(self.kv_lens[req_id])
        need_total = math.ceil((cur + n_new_tokens) / self.block_size)
        have = int((self.block_table[req_id] >= 0).sum())
        extra = need_total - have
        assert extra >= 0
        if extra:
            assert len(self._free) >= extra, "块池耗尽"
            blocks = [self._free.pop() for _ in range(extra)]
            self.block_table[req_id, have:have + extra] = torch.tensor(
                blocks, dtype=torch.int32, device=self.k.device)

    def free(self, req_id: int) -> None:
        table = self.block_table[req_id]
        used = table[table >= 0].tolist()
        self._free.extend(used)
        table[:] = -1
        self.kv_lens[req_id] = 0

    # ---- 读写 --------------------------------------------------------
    def _locate(self, req_id: int, pos: int) -> tuple[int, int]:
        logical, off = divmod(pos, self.block_size)
        blk = int(self.block_table[req_id, logical])
        assert blk >= 0, f"位置 {pos} 未分配（table=-1）"
        return blk, off

    def append(self, req_id: int, layer_idx: int, k: torch.Tensor,
               v: torch.Tensor, start: int) -> None:
        """k/v [1, n_kv, s, hd]；start 为该请求已有 token 数（须等于 kv_lens）。
        自动补块：跨块增长是常态（decode 每步 1 token）。"""
        s = k.shape[2]
        self.extend(req_id, s)
        # 教学版逐 token 写入；按段拷贝的优化版留给进阶
        for t in range(s):
            blk, off = self._locate(req_id, start + t)
            self.k[layer_idx, blk, :, off] = k[0, :, t].to(self.k.dtype)
            self.v[layer_idx, blk, :, off] = v[0, :, t].to(self.v.dtype)
        self.kv_lens[req_id] = start + s

    def gather(self, req_id: int, layer_idx: int, upto: int):
        """拼接出 [1, n_kv, upto, hd] 的连续拷贝。"""
        n_blk = math.ceil(upto / self.block_size)
        blocks = self.block_table[req_id, :n_blk].long()
        k = self.k[layer_idx][blocks]                       # [n_blk, n_kv, bs, hd]
        v = self.v[layer_idx][blocks]
        k = k.permute(1, 0, 2, 3).reshape(
            self.n_kv, n_blk * self.block_size, self.hd)[:, :upto]
        v = v.permute(1, 0, 2, 3).reshape(
            self.n_kv, n_blk * self.block_size, self.hd)[:, :upto]
        return k.unsqueeze(0).contiguous(), v.unsqueeze(0).contiguous()

    def view(self) -> CacheView:
        return CacheView(self.k, self.v, self.block_table, self.kv_lens,
                         self.block_size)
