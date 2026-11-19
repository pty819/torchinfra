import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dit_forward import QwenDiT, build_rope

def test_shapes_and_finite():
    torch.manual_seed(0)
    m = QwenDiT(num_layers=2).to(torch.float32).eval()
    B, Lt, Hl, Wl = 1, 7, 8, 8            # 8x8 latent → 4x4=16 img tokens
    ctx = torch.randn(B, Lt, 4096)
    x = torch.randn(B, 16, Hl, Wl)
    t = torch.tensor([500.0])
    pe = build_rope(Lt, Hl // 2, Wl // 2)
    with torch.no_grad():
        out = m(x, ctx, t, pe, prefix_len=Lt)
    assert out.shape == x.shape, f"{out.shape} != {x.shape}"
    assert torch.isfinite(out).all(), "输出含 NaN/Inf"
    assert out.std() > 1e-4, "输出退化（接近常数）"

def test_patchify_roundtrip():
    torch.manual_seed(1)
    x = torch.randn(2, 16, 8, 10)
    p = QwenDiT._patchify(x)
    assert p.shape == (2, 4 * 5, 64), p.shape  # [B, tokens, C*4]
    back = QwenDiT._unpatchify(p, x.shape)
    assert torch.allclose(x, back, atol=1e-6)

def test_rope_shapes():
    cos, sin = build_rope(5, 3, 4)
    assert cos.shape == (5 + 12, 64), cos.shape
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
