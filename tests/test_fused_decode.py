"""fused_qk_rope_append 差分测试：与独立 torch 参考实现逐位对拍。

参考实现按 modules.RMSNorm / batch_model.rope_decode_batch 的数学定义
独立推导（fp32 累加、rotate-half、cos/sin 取表前半区），不复用被测路径。
"""
import torch

from torchinfra.fused_decode import fused_qk_rope_append

B, NQ, NKV, D = 5, 16, 8, 128
HALF = D // 2
BLOCK = 16
EPS = 1e-6
MAX_POS = 64


def reference(qkv, qnw, knw, cos, sin, lens, bt, n_blocks):
    """独立推导：q/k per-head RMSNorm → rotate-half RoPE；k/v 写 cache。"""
    Bq = qkv.shape[0]
    q = qkv[:, :NQ * D].float().view(Bq, NQ, D)
    k = qkv[:, NQ * D:(NQ + NKV) * D].float().view(Bq, NKV, D)
    v = qkv[:, (NQ + NKV) * D:].view(Bq, NKV, D)

    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) \
        * qnw.float().view(1, 1, D)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) \
        * knw.float().view(1, 1, D)

    pos = lens.long()                                  # [B]
    c = cos[pos, :HALF].float().unsqueeze(1)           # [B,1,HALF]
    s = sin[pos, :HALF].float().unsqueeze(1)
    rope = lambda x: torch.cat(
        [x[..., :HALF] * c - x[..., HALF:] * s,
         x[..., HALF:] * c + x[..., :HALF] * s], dim=-1)
    q, k = rope(q), rope(k)

    kc = torch.full((n_blocks, NKV, BLOCK, D), float("nan"),
                    dtype=qkv.dtype, device=qkv.device)
    vc = torch.full_like(kc, float("nan"))
    for b in range(Bq):
        p = int(lens[b])
        blk, slot = int(bt[b, p // BLOCK]), p % BLOCK
        kc[blk, :, slot] = k[b].to(qkv.dtype)
        vc[blk, :, slot] = v[b]
    return q.to(qkv.dtype), kc, vc


def main():
    torch.manual_seed(0)
    dev = "cuda"
    n_blocks = 12
    qkv = (torch.randn(B, (NQ + 2 * NKV) * D, device=dev, dtype=torch.bfloat16)
           * 3)
    qnw = torch.randn(D, device=dev, dtype=torch.bfloat16)
    knw = torch.randn(D, device=dev, dtype=torch.bfloat16)
    freqs = torch.randn(MAX_POS, D // 2, device=dev)
    cos = torch.cat([freqs, freqs], dim=-1).contiguous()   # [max_pos, D] 双份
    sin = torch.cat([-freqs.flip(-1), freqs], dim=-1).contiguous()
    lens = torch.tensor([3, 16, 17, 33, 63], dtype=torch.int32, device=dev)
    bt = torch.full((B, 8), -1, dtype=torch.int32, device=dev)
    bt[:, :4] = torch.arange(1, B + 1, device=dev, dtype=torch.int32)\
        .unsqueeze(1).expand(B, 4).contiguous()   # 每请求固定块号 b+1

    q_ref, kc_ref, vc_ref = reference(qkv, qnw, knw, cos, sin, lens, bt,
                                      n_blocks)
    kc = torch.full((3, n_blocks, NKV, BLOCK, D), float("nan"),
                    dtype=torch.bfloat16, device=dev)
    vc = torch.full_like(kc, float("nan"))
    LAYER = 2     # 非 0 层：覆盖层偏移寻址（曾因此漏网：单层 cache 掩盖 bug）
    q = fused_qk_rope_append(qkv, qnw, knw, cos, sin, lens, bt, kc, vc,
                             BLOCK, layer_idx=LAYER, n_heads=NQ, n_kv=NKV,
                             eps=EPS)
    kc, vc = kc[LAYER], vc[LAYER]
    assert torch.isnan(kc[0].float()).all(), "写到了错误的层（层偏移缺失）"
    assert q.shape == (B, NQ, D)

    dq = (q.float() - q_ref.float()).abs().max().item()
    assert dq < 0.02, f"q 偏差 {dq}"

    for b in range(B):
        p = int(lens[b])
        blk, slot = int(bt[b, p // BLOCK]), p % BLOCK
        dk = (kc[blk, :, slot].float()
              - kc_ref[blk, :, slot].float()).abs().max().item()
        dv = (vc[blk, :, slot].float()
              - vc_ref[blk, :, slot].float()).abs().max().item()
        assert dk < 0.02, f"k[{b}] 偏差 {dk}"
        assert dv == 0, f"v[{b}] 偏差 {dv}"   # v 直写应零偏差

    # 只写了目标槽位，其它槽位应保持 NaN（无越界狂野写入）
    written = {(int(bt[b, int(lens[b]) // BLOCK]), int(lens[b]) % BLOCK)
               for b in range(B)}
    for blk in range(n_blocks):
        for slot in range(BLOCK):
            if (blk, slot) not in written:
                assert torch.isnan(kc[blk, :, slot].float()).all(), \
                    f"越界写入 blk={blk} slot={slot}"
    print(f"fused_qk_rope_append OK（q max diff {dq:.4f}，v 零偏差）")


if __name__ == "__main__":
    main()
