import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dit_forward import FP8OrBF16Linear

def _mk(mode):
    torch.manual_seed(0)
    w = torch.randn(4096, 4096) * 0.02
    scale = (w.abs().amax(1, keepdim=True) / 448.0).float()
    q = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return FP8OrBF16Linear(q.to("cuda"), scale.to("cuda"), mode=mode), w.to("cuda").to(torch.bfloat16)

def test_rowwise_scaled_mm_supported_and_close():
    if not torch.cuda.is_available():
        import pytest; pytest.skip("no cuda")
    lin, w_bf16 = _mk("fp8")
    assert lin.fp8_ok, "fp8 探测失败-测试走的是退化路径!"
    x = (torch.randn(128, 4096, device="cuda") * 0.02).to(torch.bfloat16)
    y_fp8 = lin(x)
    y_ref = torch.nn.functional.linear(x, w_bf16)
    rel = ((y_fp8.float() - y_ref.float()).norm() / y_ref.float().norm()).item()
    assert rel < 0.05, f"fp8 相对误差 {rel:.4f}"

def test_bf16_mode_exact():
    if not torch.cuda.is_available():
        import pytest; pytest.skip("no cuda")
    lin, w_bf16 = _mk("bf16")
    x = (torch.randn(8, 4096, device="cuda") * 0.02).to(torch.bfloat16)
    y1, y2 = lin(x), torch.nn.functional.linear(x, lin.w.to(x.dtype))
    assert torch.allclose(y1, y2)
