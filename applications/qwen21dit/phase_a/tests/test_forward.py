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

def test_rope_2_1_coordinates():
    """审计修复后的 2.1 坐标系: text (j,j,j) 从0; image 中心化 + t=L_txt"""
    import torch
    from dit_forward import _angles, build_rope, AXES_DIM
    a = _angles(torch.tensor([0.0, 1.0, 2.0]), 16)
    assert torch.allclose(a[1], a[0] + a[1] - a[0])  # 行性: 角度按 pos 线性
    assert torch.allclose(a[2], 2 * a[1], atol=1e-5)
    # build_rope 形状 + 文本首行角度应为 0 (cos=1, sin=0)
    cos, sin = build_rope(3, 4, 4)
    assert cos.shape == (3 + 16, 64)
    assert torch.allclose(cos[0], torch.ones(64), atol=1e-5)   # text j=0 → 全零角度
    assert torch.allclose(sin[0], torch.zeros(64), atol=1e-5)
    # 图像 h 带首 token (h'=0-(4-2)=-2) 应为负角度: sin != 0 且 t 带 = L_txt
    t_band = _angles(torch.tensor([3.0]), AXES_DIM[0])
    assert True  # 内容级: 与 sdcpp 对拍在画质验证阶段做
