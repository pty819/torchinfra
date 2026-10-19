"""批量 decode 路径：per-request starts 的 rope + paged attention 组装。"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_decode_batch_kernel(x_ptr, cos_ptr, sin_ptr, starts_ptr,
                              N_HEADS: tl.constexpr, D: tl.constexpr,
                              HALF: tl.constexpr, stride_row):
    row = tl.program_id(0)          # 行主序 [B, heads, 1, D]
    b = row // N_HEADS
    pos = tl.load(starts_ptr + b)
    offs = tl.arange(0, HALF)
    base = row * stride_row
    x1 = tl.load(x_ptr + base + offs).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF + offs).to(tl.float32)
    cos = tl.load(cos_ptr + pos * D + offs)
    sin = tl.load(sin_ptr + pos * D + offs)
    tl.store(x_ptr + base + offs,
             (x1 * cos - x2 * sin).to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + base + HALF + offs,
             (x2 * cos + x1 * sin).to(x_ptr.dtype.element_ty))


def _compact4d(x: torch.Tensor) -> torch.Tensor:
    """[B, H, 1, D] 的假连续陷阱同 P2 rope：强制 stride(2)==D。"""
    x = x.contiguous()
    if x.stride(2) != x.shape[-1]:
        x = x.clone(memory_format=torch.contiguous_format)
    return x


def rope_decode_batch(q: torch.Tensor, k: torch.Tensor,
                      starts: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """原地旋转 q/k [B, heads, 1, D]；starts 为各请求位置（int32）。"""
    q, k = _compact4d(q), _compact4d(k)
    d = q.shape[-1]
    for x in (q, k):
        heads = x.shape[1]
        rows = x.numel() // d
        _rope_decode_batch_kernel[(rows,)](
            x, cos, sin, starts, heads, d, d // 2, x.stride(2))
    return q, k


@torch.no_grad()
def decode_step(model, token_ids: torch.Tensor, paged, req_ids: list[int]):
    """批量 decode 一步：token_ids [B,1] → logits [B, vocab]。

    attention 走 paged kernel；norm/MLP 等复用 model 既有层。
    """
    from torchinfra.paged_attn import paged_decode_attention
    assert token_ids.shape[1] == 1
    B = token_ids.shape[0]
    starts = paged.kv_lens[req_ids].to(torch.int32).clone()  # append 前的位置
    rope_cos, rope_sin = model.rope.cos, model.rope.sin
    x = model.model.embed_tokens(token_ids)
    for i, layer in enumerate(model.model.layers):
        h = layer.input_layernorm(x)
        attn = layer.self_attn
        q = attn.q_proj(h).view(B, 1, attn.n_heads, attn.hd).transpose(1, 2)
        k = attn.k_proj(h).view(B, 1, attn.n_kv, attn.hd).transpose(1, 2)
        v = attn.v_proj(h).view(B, 1, attn.n_kv, attn.hd).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = rope_decode_batch(q, k, starts, rope_cos, rope_sin)
        for bi, r in enumerate(req_ids):
            paged.append(r, i, k[bi:bi + 1], v[bi:bi + 1], int(starts[bi]))
        view = paged.view().slice_batch(req_ids)   # append 后再切片（kv_lens 已更新）
        out = paged_decode_attention(q, view, layer_idx=i)      # [B, H, D]
        x = x + attn.o_proj(out.reshape(B, 1, attn.n_heads * attn.hd))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    x = model.model.norm(x)
    return model.lm_head(x[:, -1:])[:, 0, :]       # [B, vocab]
