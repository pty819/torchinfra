# tests/test_batch_model.py
import torch
from torchinfra.triton_kernels import TritonRoPE
from torchinfra.batch_model import rope_decode_batch


def test_rope_batch():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        ref = TritonRoPE(128, 512, 1e6).cuda()
        B, Hq, Hk, D = 4, 16, 8, 128
        starts = torch.tensor([7, 0, 33, 1], dtype=torch.int32, device="cuda")
        q = torch.randn(B, Hq, 1, D, device="cuda", dtype=dtype)
        k = torch.randn(B, Hk, 1, D, device="cuda", dtype=dtype)
        q2, k2 = q.clone(), k.clone()
        rope_decode_batch(q2, k2, starts, ref.cos, ref.sin)
        for b in range(B):
            qr, kr = ref.apply_qk(q[b:b + 1].clone(), k[b:b + 1].clone(),
                                  int(starts[b]))
            torch.testing.assert_close(q2[b].float(), qr[0].float(),
                                       rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(k2[b].float(), kr[0].float(),
                                       rtol=1e-2, atol=1e-2)
    print("rope_decode_batch OK")


def test_decode_step():
    from tests._ref import MODEL_DIR, PROMPT
    from torchinfra.model import Qwen3ForCausalLM
    from torchinfra.kv_cache import KVCache
    from torchinfra.paged_cache import PagedKVCache, MirroredKVCache
    from torchinfra.batch_model import decode_step
    from torchinfra.tokenizer import TokenizerBox

    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    cfg = model.cfg
    tok = TokenizerBox(MODEL_DIR)
    prompts = ["你好，世界。", "解释一下RoPE。", "1+1="]
    req_ids = [0, 1, 2]
    lens = []
    paged = PagedKVCache(cfg.num_hidden_layers, cfg.num_key_value_heads,
                         cfg.head_dim, num_blocks=96, block_size=16,
                         dtype=torch.float32)
    with torch.no_grad():
        for r, p in zip(req_ids, prompts):
            ids = tok.encode(p)
            mirror = MirroredKVCache(paged, req_id=r, max_seq=len(ids))
            logits = model(torch.tensor([ids], device="cuda"), kv=mirror,
                           start_pos=0, logits_to_keep="last")
            lens.append(len(ids))
        # P1 参考路径：contiguous cache 重建 + 单 token forward
        next_tokens = [10, 20, 30]
        refs = []
        for r, p, nt in zip(req_ids, prompts, next_tokens):
            ids = tok.encode(p)
            kv1 = KVCache(cfg.num_hidden_layers, len(ids) + 8,
                          cfg.num_key_value_heads, cfg.head_dim,
                          dtype=torch.float32)
            model(torch.tensor([ids], device="cuda"), kv=kv1, start_pos=0,
                  logits_to_keep="last")
            with torch.no_grad():
                ref = model(torch.tensor([[nt]], device="cuda"), kv=kv1,
                            start_pos=len(ids), logits_to_keep="last")
            refs.append(ref[0, -1])
        # batched 路径
        out = decode_step(model,
                          torch.tensor([next_tokens], device="cuda").t(),  # [3,1]
                          paged, req_ids)
        for i in range(3):
            torch.testing.assert_close(out[i], refs[i], rtol=1e-3, atol=1e-3)
    print("decode_step OK")


def main():
    test_rope_batch()
    test_decode_step()


if __name__ == "__main__":
    main()
