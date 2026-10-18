"""JIT 编译并加载 torchinfra 的自定义 HIP kernel。

环境约定（详见 README）：
- ROCM_HOME 无需手动设置：torch 自动解析到 venv 内的 _rocm_sdk_core（hipcc 所在地）
- PYTORCH_ROCM_ARCH 显式固定为 gfx1201（Radeon RX 9070 GRE），
  避免对"所有可见卡"编译并消除 warning
- pip 版 ROCm 下 clang 找不到 device bitcode（布局与 /opt/rocm 不同），
  需显式传 --rocm-device-lib-path
"""

import os
from pathlib import Path

_SRC_DIR = Path(__file__).parent / "src"

_module = None


def _device_lib_flag() -> str:
    from torch.utils.cpp_extension import ROCM_HOME

    bitcode = Path(ROCM_HOME) / "lib" / "llvm" / "amdgcn" / "bitcode"
    if not bitcode.is_dir():
        raise RuntimeError(f"未找到 ROCm device bitcode: {bitcode}")
    return f"--rocm-device-lib-path={bitcode}"


def load_kernels(verbose: bool = False) -> None:
    """编译（或命中缓存后加载）kernel 库并注册 torch.ops.torchinfra.*；幂等。

    只用 TORCH_LIBRARY 注册算子、不导出 Python 模块，故 is_python_module=False。
    """
    global _module
    if _module is not None:
        return
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")
    from torch.utils.cpp_extension import load

    load(
        name="torchinfra_kernels",
        sources=[str(_SRC_DIR / "vector_add.cu")],
        # ROCm 构建下 extra_cuda_cflags 会被传给 hipcc
        extra_cuda_cflags=["-O2", _device_lib_flag()],
        verbose=verbose,
        is_python_module=False,
    )
    _module = True
