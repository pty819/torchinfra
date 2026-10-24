# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Trimmed vendoring of ROCm/aiter main: aiter/ops/flydsl/kernels/tensor_shim.py
# Kept ONLY what flash_attn_func_gfx1201.py imports (`_run_compiled`); the
# TensorView/GTensor/STensor classes were dropped because they pull in
# `flydsl.expr.vector`, which the 0.4.1.dev993 wheel does not ship.

import flydsl.compiler as flyc


def _run_compiled(exe, *args, **kwargs):
    """First call: ``flyc.compile(exe, *args, **kwargs)`` compiles **and**
    executes the kernel. Subsequent calls: fast dispatch via the cached
    ``CompiledFunction`` (which takes positional args only — stream routing
    on this path follows the stream captured at compile time / current)."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args, **kwargs)
        exe._cf = cf
    else:
        # cached fast path: the generated dispatch indexes all kernel params
        # positionally (incl. the stream slot), so append stream as the last
        # positional arg instead of a kwarg
        cf(*args, kwargs.pop("stream", None))
