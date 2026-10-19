"""decode 路径 CUDA Graph 化：GPU 侧 cache 写入 kernel + 整步图捕获（vLLM 式）。

三件事消灭 decode 的 CPU 开销：
1. append_kv kernel：cache 写入全在 GPU（索引来自 GPU 张量，无 int() 同步）
2. 静态缓冲 + pad：token/block_table/kv_lens 固定形状缓冲，pad 槽位指向预留的
   dummy 块 0（写入被丢弃，attention 读到零值、结果被丢弃）
3. torch.cuda.CUDAGraph 捕获整个 decode step：一次 replay 替代数百次 kernel launch
"""
import torch
import triton
import triton.language as tl

from torchinfra.paged_attn import paged_decode_attention_tensors

@triton.jit
def _append_kv_kernel(K_cache, V_cache, K_new, V_new, block_table, kv_lens,
                      s_kc_blk, s_kc_h,
                      s_kn_b, s_kn_h, s_vn_b, s_vn_h,
                      s_bt_b, s_bt_s,
                      HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(kv_lens + b)                       # GPU 读取，无同步
    blk = tl.load(block_table + b * s_bt_b
                  + (pos // BLOCK) * s_bt_s).to(tl.int64)
    blk = tl.maximum(blk, 0)   # 未分配槽位是 -1：钳到 dummy 块 0，防狂野写入
    slot = pos % BLOCK
    offs_d = tl.arange(0, HEAD_DIM)
    kn = tl.load(K_new + b * s_kn_b + h * s_kn_h + offs_d)
    vn = tl.load(V_new + b * s_vn_b + h * s_vn_h + offs_d)
    tl.store(K_cache + blk * s_kc_blk + h * s_kc_h + slot * HEAD_DIM + offs_d,
             kn.to(K_cache.dtype.element_ty))
    tl.store(V_cache + blk * s_kc_blk + h * s_kc_h + slot * HEAD_DIM + offs_d,
             vn.to(V_cache.dtype.element_ty))


def append_kv(k_cache: torch.Tensor, v_cache: torch.Tensor,
              k_new: torch.Tensor, v_new: torch.Tensor,
              block_table: torch.Tensor, kv_lens: torch.Tensor,
              block_size: int, layer_idx: int = 0) -> None:
    """把 k_new/v_new [B, n_kv, 1, D] 写入各请求 kv_len 指示的槽位。

    k/v_new 允许 transposed 视图，只要 head_dim 维连续（按 stride 寻址）。
    不更新 kv_lens（由调用方统一管理，图内为 add_(1)）。
    """
    kc = k_cache[layer_idx]
    vc = v_cache[layer_idx]
    B, n_kv = k_new.shape[0], k_new.shape[1]
    _append_kv_kernel[(B, n_kv)](
        kc, vc, k_new, v_new, block_table, kv_lens,
        kc.stride(0), kc.stride(1),
        k_new.stride(0), k_new.stride(1),
        v_new.stride(0), v_new.stride(1),
        block_table.stride(0), block_table.stride(1),
        HEAD_DIM=k_new.shape[-1], BLOCK=block_size)


class DecodeGraphRunner:
    """整步 decode 的 CUDA Graph 捕获与回放（vLLM 式）。

    - 静态缓冲：tokens [B,1]、bt [B, W]（pad 行全 0 → dummy 块）、lens/lens_p1 [B]
    - rope 与 append 读 lens（=start）；attention 读 lens_p1（=start+1）。
      两个独立缓冲而非图内 add_：层间共享同一 start，原地自增会被 28 层累加
    - pad 槽位 lens=1：attention 读 dummy 块的零值，结果丢弃
    """

    def __init__(self, model, paged, batch_size: int):
        self.model = model
        self.paged = paged
        self.B = batch_size
        W = paged.block_table.shape[1]
        dev = paged.k.device
        self.tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=dev)
        self.bt = torch.zeros(batch_size, W, dtype=torch.int32, device=dev)
        self.lens = torch.ones(batch_size, dtype=torch.int32, device=dev)
        self.lens_p1 = torch.ones(batch_size, dtype=torch.int32, device=dev)

        # 标准 CUDA Graph 捕获协议：旁路流预热 3 次，再捕获
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._step()

    def _step(self) -> torch.Tensor:
        from torchinfra.batch_model import rope_decode_batch
        model, paged = self.model, self.paged
        B = self.B
        cos, sin = model.rope.cos, model.rope.sin
        x = model.model.embed_tokens(self.tokens)
        from torchinfra.fuse import attn_qkv, mlp_forward
        for i, layer in enumerate(model.model.layers):
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q, k, v = attn_qkv(h, attn)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            q, k = rope_decode_batch(q, k, self.lens, cos, sin)
            append_kv(paged.k, paged.v, k, v, self.bt, self.lens,
                      paged.block_size, layer_idx=i)
            out = paged_decode_attention_tensors(
                q, paged.k, paged.v, self.bt, self.lens_p1, i,
                paged.block_size)
            x = x + attn.o_proj(out.reshape(B, 1, attn.n_heads * attn.hd))
            x = x + mlp_forward(layer.post_attention_layernorm(x), layer.mlp)
        x = model.model.norm(x)
        return model.lm_head(x[:, -1:])[:, 0, :]

    def run(self, tokens: list[int], rows: list[tuple[int, int]]
            ) -> torch.Tensor:
        """rows: [(rid, 当前 kv_len)]；回放并返回 logits [len(rows), vocab]。

        返回值是图输出缓冲的视图，须在下一次 replay 前消费。
        """
        n = len(rows)
        assert n <= self.B, f"batch {n} 超过捕获档位 {self.B}"
        self.tokens.zero_()
        self.bt.zero_()
        self.lens.fill_(1)
        self.tokens[:n, 0] = torch.tensor(tokens, dtype=torch.long,
                                           device=self.tokens.device)
        for i, (rid, cur) in enumerate(rows):
            self.bt[i].copy_(self.paged.block_table[rid])
            self.lens[i] = cur
        self.lens_p1.copy_(self.lens)
        self.lens_p1 += 1
        self.graph.replay()
        return self.out[:n]


# ---- piecewise（Inductor cudagraph trees + custom op 边界切图）------------
_OPS_REGISTERED = False


def _register_ops():
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return
    from torch.library import custom_op
    from torchinfra.batch_model import rope_decode_batch

    @custom_op("torchinfra::rope_op", mutates_args=("q", "k"))
    def rope_op(q: torch.Tensor, k: torch.Tensor, starts: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> None:
        q2, k2 = rope_decode_batch(q, k, starts, cos, sin)
        if q2 is not q:
            q.copy_(q2)
        if k2 is not k:
            k.copy_(k2)

    @custom_op("torchinfra::append_op", mutates_args=("k_cache", "v_cache"))
    def append_op(k_cache: torch.Tensor, v_cache: torch.Tensor,
                  k_new: torch.Tensor, v_new: torch.Tensor, bt: torch.Tensor,
                  lens: torch.Tensor, block_size: int,
                  layer_idx: int) -> None:
        append_kv(k_cache, v_cache, k_new, v_new, bt, lens, block_size,
                  layer_idx)

    @custom_op("torchinfra::attn_op", mutates_args=())
    def attn_op(q: torch.Tensor, k_full: torch.Tensor, v_full: torch.Tensor,
                bt: torch.Tensor, lens_p1: torch.Tensor, layer_idx: int,
                block_size: int) -> torch.Tensor:
        return paged_decode_attention_tensors(q, k_full, v_full, bt, lens_p1,
                                              layer_idx, block_size)

    @attn_op.register_fake
    def _(q, k_full, v_full, bt, lens_p1, layer_idx, block_size):
        return torch.empty(q.shape[0], q.shape[1], q.shape[3],
                           dtype=q.dtype, device=q.device)

    _OPS_REGISTERED = True


class DecodeGraphRunnerPiecewise:
    """piecewise CUDA Graph：torch.compile(mode=reduce-overhead)。

    与手动整图同缓冲同语义；Inductor 额外融合 custom op 之间的
    norm/proj/MLP 链，实测比手动整图再快 ~1.4x（图内 kernel 数更少）。
    """

    def __init__(self, model, paged, batch_size: int):
        _register_ops()
        self.model = model
        self.paged = paged
        self.B = batch_size
        W = paged.block_table.shape[1]
        dev = paged.k.device
        self.tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=dev)
        self.bt = torch.zeros(batch_size, W, dtype=torch.int32, device=dev)
        self.lens = torch.ones(batch_size, dtype=torch.int32, device=dev)
        self.lens_p1 = torch.ones(batch_size, dtype=torch.int32, device=dev)
        self._fn = torch.compile(self._step, mode="reduce-overhead")
        # 预热（编译 + cudagraph 捕获；dummy 状态写入 dummy 块）
        with torch.no_grad():
            for _ in range(3):
                self._fn()

    def _step(self) -> torch.Tensor:
        torchinfra_rope = torch.ops.torchinfra.rope_op
        torchinfra_append = torch.ops.torchinfra.append_op
        torchinfra_attn = torch.ops.torchinfra.attn_op
        model, paged = self.model, self.paged
        B = self.B
        cos, sin = model.rope.cos, model.rope.sin
        x = model.model.embed_tokens(self.tokens)
        from torchinfra.fuse import attn_qkv, mlp_forward
        for i, layer in enumerate(model.model.layers):
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q, k, v = attn_qkv(h, attn)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            torchinfra_rope(q, k, self.lens, cos, sin)
            torchinfra_append(paged.k, paged.v, k, v, self.bt, self.lens,
                              paged.block_size, i)
            out = torchinfra_attn(q, paged.k, paged.v, self.bt, self.lens_p1,
                                  i, paged.block_size)
            x = x + attn.o_proj(out.reshape(B, 1, attn.n_heads * attn.hd))
            x = x + mlp_forward(layer.post_attention_layernorm(x), layer.mlp)
        x = model.model.norm(x)
        return model.lm_head(x[:, -1:])[:, 0, :]

    def run(self, tokens: list[int], rows: list[tuple[int, int]]
            ) -> torch.Tensor:
        """与 DecodeGraphRunner.run 同语义（返回图缓冲视图，replay 前消费）。"""
        n = len(rows)
        assert n <= self.B
        self.tokens.zero_()
        self.bt.zero_()
        self.lens.fill_(1)
        self.tokens[:n, 0] = torch.tensor(tokens, dtype=torch.long,
                                           device=self.tokens.device)
        for i, (rid, cur) in enumerate(rows):
            self.bt[i].copy_(self.paged.block_table[rid])
            self.lens[i] = cur
        self.lens_p1.copy_(self.lens)
        self.lens_p1.add_(1)
        with torch.no_grad():
            return self._fn()[:n]
