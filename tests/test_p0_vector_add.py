"""P0 验收测试：自定义 HIP kernel 编译链路 + bf16 vector_add 正确性。

运行方式（仓库根目录）：
    PYTHONPATH=. ~/rocm10/bin/python tests/test_p0_vector_add.py

首次运行会现场编译 kernel（可能需要一两分钟）；后续运行走缓存。
"""

import torch


def main() -> None:
    assert torch.cuda.is_available(), "需要可用 GPU"
    assert torch.version.hip is not None, "需要 ROCm 版 PyTorch"

    from torchinfra.kernels import load_kernels

    load_kernels()
    op = torch.ops.torchinfra.vector_add

    # 用例 1：1M 元素 bf16，大数组对拍（覆盖多 block）
    n = 1 << 20
    a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = op(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float(), a.float() + b.float(), rtol=1e-2, atol=1e-2
    )

    # 用例 2：长度为质数（1000003），覆盖尾部不完整 block 的边界
    n2 = 1_000_003
    a2 = torch.randn(n2, device="cuda", dtype=torch.bfloat16)
    b2 = torch.randn(n2, device="cuda", dtype=torch.bfloat16)
    out2 = op(a2, b2)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out2.float(), a2.float() + b2.float(), rtol=1e-2, atol=1e-2
    )

    # 用例 3：空张量不崩溃（边界）
    empty = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    out3 = op(empty, empty)
    torch.cuda.synchronize()
    assert out3.numel() == 0

    print("P0 vector_add: 全部用例通过")


if __name__ == "__main__":
    main()
