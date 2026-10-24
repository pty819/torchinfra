# torchinfra/flydsl_kernels — vendored aiter FlyDSL kernels for gfx1201.
#
# Prereqs: flydsl>=0.4.1.dev993 wheel (installed via tuna mirror).
# The wheel lacks flydsl.expr.buffer_ops (post-dev993 API); flydsl_compat
# injects a vendored copy before the kernel modules are imported.
#
# flydsl's own arch detection falls back to gfx942 when rocm_agent_enumerator
# is absent — pin the arch explicitly.

import os

os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1201")

# TheRock pip SDK (rocm_sdk_devel) ships a full AMD LLVM 24.0git toolchain
# under site-packages/_rocm_sdk_devel. MLIR's HSACO serialization locates
# ld.lld via ROCM_PATH/ROCM_HOME — unset on a bare pip install, which makes
# every "gpu-module-to-binary" fail with a generic "lld invocation failed".
# Point both at the in-venv SDK and hand the same LLVM to flydsl's external
# codegen path (its bundled in-process lld hits the same lookup).
import sysconfig

_sdk_devel = os.path.join(sysconfig.get_paths()["purelib"], "_rocm_sdk_devel")
if os.path.isdir(os.path.join(_sdk_devel, "lib", "llvm", "bin")):
    os.environ.setdefault("ROCM_PATH", _sdk_devel)
    os.environ.setdefault("ROCM_HOME", _sdk_devel)
    os.environ.setdefault("FLYDSL_COMPILE_LLVM_DIR", os.path.join(_sdk_devel, "lib", "llvm"))

from . import flydsl_compat  # noqa: F401,E402  (must run before kernel import)
from .flash_attn_func_gfx1201 import build_flash_attn_func_module  # noqa: E402,F401
from .wrapper import flydsl_flash_attn  # noqa: E402,F401
