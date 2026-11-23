# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Compat shim: inject the vendored buffer_ops implementation into the
# flydsl.expr namespace so that `from flydsl.expr import buffer_ops`
# (as written in aiter's gfx1201 kernels) resolves against the published
# flydsl 0.4.1.dev993 wheel, which does not ship that module yet.
#
# Must be imported BEFORE importing any aiter flydsl kernel module.

import sys

from . import _buffer_ops_impl


def _install_buffer_ops() -> None:
    if "flydsl.expr.buffer_ops" in sys.modules:
        return
    import flydsl.expr as _expr

    sys.modules["flydsl.expr.buffer_ops"] = _buffer_ops_impl
    _expr.buffer_ops = _buffer_ops_impl


def _strip_wave64_opt() -> None:
    """0.4.1.dev993 wheel inconsistency: its Python side passes a `wave64`
    option to the `rocdl-attach-target` MLIR pass, but the bundled MLIR libs
    reject it ("no such option wave64"). Strip it from the pass options here
    instead of patching site-packages. Wave size still derives from
    chip=gfx1201 in convert-gpu-to-rocdl."""
    from flydsl.compiler.backends.rocm import RocmBackend

    if getattr(RocmBackend._format_pass_opts, "_torchinfra_stripped", False):
        return
    _orig = RocmBackend._format_pass_opts

    def _fmt(opts: dict) -> str:
        opts = {k: v for k, v in opts.items() if k != "wave64"}
        return _orig(opts)

    _fmt._torchinfra_stripped = True
    RocmBackend._format_pass_opts = staticmethod(_fmt)


def _generic_asm_bridge() -> None:
    """Bundled MLIR is 24.0git; the external mlir-opt we can get (Debian 23)
    rejects 24's pretty syntax (e.g. `fastmath<fast>` properties). MLIR 24's
    *generic* form prints the legacy attribute spelling (`fastmathFlags =
    #llvm.fastmath<fast>`) which 23 parses fine — verified empirically. Scope a
    print_generic_op_form=True get_asm override to the external-codegen file
    write only, so nothing else (dumps, cache keys) changes format."""
    import flydsl.compiler.external_llvm as ext
    from flydsl._mlir import ir

    if getattr(ext.run_external_binary_codegen, "_torchinfra_bridged", False):
        return
    _orig_run = ext.run_external_binary_codegen
    _orig_get_asm = ir.Operation.get_asm

    def _get_asm_generic(self, *a, **k):
        k.setdefault("print_generic_op_form", True)
        txt = _orig_get_asm(self, *a, **k)
        # wheel-bundled MLIR prints `#rocdl.wmma_c_modifier<none>`, which
        # TheRock's parser rejects; the property is optional (defaults to
        # none) — verified empirically — so strip it from the emitted text.
        txt = txt.replace("modC = #rocdl.wmma_c_modifier<none>, ", "")
        txt = txt.replace(", modC = #rocdl.wmma_c_modifier<none>", "")
        # MLIR 24 prints llvm.mlir.constant with an index-typed value while
        # the result is i64; the external MLIR 23 parser rejects that pair.
        # The value is an integer constant, so spell it with the result type.
        import re
        txt = re.sub(
            r'("llvm.mlir.constant"\(\) <\{value = [^}]+) : index(\}> : \(\) -> (i\d+))',
            lambda m: f"{m.group(1)} : {m.group(3)}{m.group(2)}",
            txt,
        )
        return txt

    def _run(*a, **k):
        # keep intermediate files for debugging (external_llvm cleans its
        # TemporaryDirectory on the way out); work_dir must be a Path
        from pathlib import Path

        k.setdefault("work_dir", Path("/tmp/flydsl_ext_debug"))
        ir.Operation.get_asm = _get_asm_generic
        try:
            return _orig_run(*a, **k)
        finally:
            ir.Operation.get_asm = _orig_get_asm

    _run._torchinfra_bridged = True
    ext.run_external_binary_codegen = _run


_install_buffer_ops()
_strip_wave64_opt()
_generic_asm_bridge()
