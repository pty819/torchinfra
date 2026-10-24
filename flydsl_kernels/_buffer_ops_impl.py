# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Vendored from ROCm/FlyDSL main: python/flydsl/expr/buffer_ops.py
# (post-0.4.1.dev993 API — the published wheel lacks flydsl.expr.buffer_ops,
#  but aiter's gfx1201 kernels import it from there).
# Only change: relative imports rewritten to absolute so this file can live
# outside the flydsl package. Injected into the flydsl.expr namespace by
# flydsl_compat.py. Do not edit the logic — keep diffable against upstream.

"""AMD Buffer Load/Store Operations - High-level Python API

This module provides high-level Python wrappers for AMD CDNA3/CDNA4 buffer operations.
Buffer operations use a scalar base pointer and per-thread offsets for efficient memory access.
"""

from typing import Optional, Union

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm, rocdl
from flydsl._mlir.extras import types as T
from flydsl.runtime.device import is_rdna_arch
from flydsl.expr.meta import dsl_loc_tracing


def _get_buffer_flags(arch=None):
    """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

    Constructs the 32-bit flags field for rocdl.make.buffer.rsrc, following the
    same logic as LLVM's AMDGPUToROCDL makeBufferRsrc().

    CDNA (gfx9xx):    (7 << 12) | (4 << 15)                         = 0x20070
    RDNA (gfx10+):    (7 << 12) | (4 << 15) | (1 << 24) | (2 << 28) = 0x21020070
    """

    import os

    if arch is None:
        arch = os.environ.get("FLYDSL_GPU_ARCH")
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


__all__ = [
    "create_llvm_ptr",
    "get_element_ptr",
    "create_buffer_resource",
    "create_buffer_resource_from_addr",
    "buffer_load",
    "buffer_store",
    "BufferResourceDescriptor",
    "extract_base_index",
]


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


@dsl_loc_tracing
def _create_i32_constant(value: int) -> ir.Value:
    i32_type = T.i32()
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    attr = ir.IntegerAttr.get(i32_type, value)
    op = std_arith.ConstantOp(i32_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def _create_i16_constant(value: int) -> ir.Value:
    i16_type = T.i16()
    attr = ir.IntegerAttr.get(i16_type, value)
    op = std_arith.ConstantOp(i16_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def _create_i64_constant(value: int) -> ir.Value:
    i64_type = T.i64()
    attr = ir.IntegerAttr.get(i64_type, value)
    op = std_arith.ConstantOp(i64_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def create_llvm_ptr(value, address_space: int = 0) -> ir.Value:
    """Create an LLVM pointer from an integer or index value."""
    value = _unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        i64_type = T.i64()
        value = _unwrap_value(std_arith.IndexCastOp(i64_type, value).result)
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return llvm.IntToPtrOp(ptr_type, value).result


@dsl_loc_tracing
def extract_base_index(tensor, address_space: int = 1) -> ir.Value:
    """Extract the base address of a fly.memref as an index value."""
    from flydsl._mlir.dialects import fly as _fly
    from flydsl._mlir.dialects import memref as _memref

    raw = _unwrap_value(tensor)
    try:
        ir.MemRefType(raw.type)
        return _memref.extract_aligned_pointer_as_index(raw)
    except ValueError:
        pass

    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw)
    i64_val = llvm.PtrToIntOp(ir.IntegerType.get_signless(64), ptr).result
    return _unwrap_value(std_arith.IndexCastOp(ir.IndexType.get(), i64_val).result)


@dsl_loc_tracing
def get_element_ptr(
    base_ptr,
    byte_offset: Union[int, ir.Value, None] = None,
    static_byte_offset: int = 0,
    elem_type: Optional[ir.Type] = None,
    no_wrap_flags=None,
) -> ir.Value:
    """Build an LLVM GEP from a base pointer plus byte offsets."""
    _gep_dynamic_index_sentinel = -(2**31)

    base_ptr = _unwrap_value(base_ptr)
    if not isinstance(static_byte_offset, int):
        raise TypeError(f"static_byte_offset must be int, got {type(static_byte_offset).__name__}")
    if elem_type is None:
        elem_type = T.i8()
    elif callable(elem_type):
        elem_type = elem_type()

    if byte_offset is None:
        dynamic_indices = []
        raw_constant_indices = [int(static_byte_offset)]
    elif isinstance(byte_offset, int):
        dynamic_indices = []
        raw_constant_indices = [int(byte_offset) + int(static_byte_offset)]
    else:
        offset_val = _unwrap_value(byte_offset)
        if isinstance(offset_val.type, ir.IndexType):
            i64_type = T.i64()
            offset_val = _unwrap_value(std_arith.IndexCastOp(i64_type, offset_val).result)
        elif not isinstance(offset_val.type, ir.IntegerType):
            raise TypeError("byte_offset must be int, index, or integer-typed MLIR value; " f"got {offset_val.type}")

        if static_byte_offset != 0:
            static_type = offset_val.type
            static_attr = ir.IntegerAttr.get(static_type, int(static_byte_offset))
            static_const = _unwrap_value(std_arith.ConstantOp(static_type, static_attr).result)
            offset_val = _unwrap_value(std_arith.AddIOp(offset_val, static_const).result)

        dynamic_indices = [offset_val]
        raw_constant_indices = [_gep_dynamic_index_sentinel]

    return llvm.GEPOp(
        base_ptr.type,
        base_ptr,
        dynamic_indices,
        raw_constant_indices,
        elem_type,
        no_wrap_flags,
    ).result


class BufferResourceDescriptor:
    """AMD Buffer Resource Descriptor"""

    def __init__(self, rsrc: ir.Value):
        self.rsrc = rsrc

    @staticmethod
    @dsl_loc_tracing
    def from_memref(
        memref_val: ir.Value,
        stride: int = 0,
        max_size: bool = True,
        data_format: str = "f32",
        num_records_bytes: Optional[Union[int, ir.Value]] = None,
        base_byte_offset: Optional[Union[int, ir.Value]] = None,
    ) -> "BufferResourceDescriptor":
        raw_val = _unwrap_value(memref_val)
        from flydsl._mlir.dialects import fly as _fly

        ptr_type = ir.Type.parse("!llvm.ptr")
        base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw_val)
        if base_byte_offset is not None:
            base_ptr = get_element_ptr(base_ptr, byte_offset=base_byte_offset)

        flags_val = _get_buffer_flags()
        flags = _create_i32_constant(flags_val)
        stride_val = _create_i16_constant(stride)

        def _num_records_from_memref_type() -> Optional[int]:
            try:
                mt = ir.MemRefType(_unwrap_value(memref_val).type)
                shape = list(mt.shape)
                if any(int(d) < 0 for d in shape):
                    return None
                elem_t = mt.element_type
                elem_bits = getattr(elem_t, "width", None)
                if elem_bits is None:
                    return None
                elem_bytes = int(elem_bits) // 8
                if elem_bytes <= 0:
                    return None
                num_elems = 1
                for d in shape:
                    num_elems *= int(d)
                return int(num_elems) * int(elem_bytes)
            except Exception:
                return None

        if num_records_bytes is not None:
            if isinstance(num_records_bytes, int):
                nbytes = int(num_records_bytes)
                if nbytes <= 0:
                    nbytes = 0
                if nbytes > 0xFFFFFFFF:
                    nbytes = 0xFFFFFFFF
                num_records = _create_i64_constant(nbytes)
            else:
                v = _unwrap_value(num_records_bytes)
                i64_type = T.i64()
                if not isinstance(v.type, ir.IntegerType) or v.type.width != 64:
                    if isinstance(v.type, ir.IndexType):
                        op = std_arith.IndexCastOp(i64_type, v)
                    else:
                        op = std_arith.ExtSIOp(i64_type, v)
                    v = _unwrap_value(op.result)
                num_records = v
        elif max_size:
            num_records = _create_i64_constant(0xFFFFFFFF)
        else:
            nbytes = _num_records_from_memref_type()
            if nbytes is None:
                num_records = _create_i64_constant(0xFFFFFFFF)
            else:
                if nbytes > 0xFFFFFFFF:
                    nbytes = 0xFFFFFFFF
                num_records = _create_i64_constant(int(nbytes))

        rsrc_type = ir.Type.parse("!llvm.ptr<8>")
        rsrc = rocdl.MakeBufferRsrcOp(rsrc_type, base_ptr, stride_val, num_records, flags).result

        return BufferResourceDescriptor(rsrc)


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value,
    *,
    num_records_bytes: Optional[Union[int, ir.Value]] = None,
) -> ir.Value:
    addr_i64 = _unwrap_value(addr_i64)
    ptr_type = ir.Type.parse("!llvm.ptr")
    base_ptr = llvm.IntToPtrOp(ptr_type, addr_i64).result
    flags = _create_i32_constant(_get_buffer_flags())
    stride = _create_i16_constant(0)
    if num_records_bytes is None:
        num_records = _create_i64_constant(0xFFFFFFFF)
    elif isinstance(num_records_bytes, int):
        nbytes = max(0, min(int(num_records_bytes), 0xFFFFFFFF))
        num_records = _create_i64_constant(nbytes)
    else:
        num_records = _unwrap_value(num_records_bytes)
        i64_type = T.i64()
        if not isinstance(num_records.type, ir.IntegerType) or num_records.type.width != 64:
            if isinstance(num_records.type, ir.IndexType):
                num_records = _unwrap_value(std_arith.IndexCastOp(i64_type, num_records).result)
            else:
                num_records = _unwrap_value(std_arith.ExtSIOp(i64_type, num_records).result)
    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return rocdl.MakeBufferRsrcOp(rsrc_type, base_ptr, stride, num_records, flags).result


@dsl_loc_tracing
def create_buffer_resource(
    memref_val: ir.Value,
    stride: int = 0,
    max_size: bool = True,
    *,
    num_records_bytes: Optional[Union[int, ir.Value]] = None,
    base_byte_offset: Optional[Union[int, ir.Value]] = None,
) -> ir.Value:
    desc = BufferResourceDescriptor.from_memref(
        memref_val,
        stride,
        max_size,
        num_records_bytes=num_records_bytes,
        base_byte_offset=base_byte_offset,
    )
    return desc.rsrc


@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: Optional[ir.Value] = None,
    cache_modifier: int = 0,
    soffset_bytes: Optional[Union[int, ir.Value]] = None,
) -> ir.Value:
    if dtype is None:
        dtype = T.f32()
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    offset = _unwrap_value(offset)

    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        op = std_arith.IndexCastOp(T.i32(), offset)
        offset = _unwrap_value(op.result)

    element_bytes = dtype.width // 8
    bytes_const = _create_i32_constant(element_bytes)
    op = std_arith.MulIOp(offset, bytes_const)
    offset = _unwrap_value(op.result)

    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)

    if vec_width == 1:
        result_type = dtype
    else:
        result_type = ir.VectorType.get([vec_width], dtype)

    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(soffset_bytes)
        else:
            soffset = _unwrap_value(soffset_bytes)
            if not isinstance(soffset.type, ir.IntegerType) or soffset.type.width != 32:
                op = std_arith.IndexCastOp(T.i32(), soffset)
                soffset = _unwrap_value(op.result)
    aux_flags = _create_i32_constant(cache_modifier)

    load_op = rocdl.RawPtrBufferLoadOp(
        result_type, rsrc, offset, soffset, aux_flags
    )

    return load_op.result


@dsl_loc_tracing
def buffer_store(
    data: ir.Value,
    rsrc: ir.Value,
    offset: ir.Value,
    mask: Optional[ir.Value] = None,
    cache_modifier: int = 0,
    *,
    soffset_bytes: Optional[Union[int, ir.Value]] = None,
    offset_is_bytes: bool = False,
):
    if hasattr(data, "ir_value"):
        data = data.ir_value()
    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = _unwrap_value(offset)

    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        op = std_arith.IndexCastOp(T.i32(), offset)
        offset = _unwrap_value(op.result)

    if not offset_is_bytes:
        data_type = data.type
        if hasattr(data_type, "element_type"):
            element_type = data_type.element_type
        else:
            element_type = data_type
        element_bytes = element_type.width // 8
        bytes_const = _create_i32_constant(element_bytes)
        op = std_arith.MulIOp(offset, bytes_const)
        offset = _unwrap_value(op.result)

    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)

    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(int(soffset_bytes))
        else:
            soffset = _unwrap_value(soffset_bytes)
            if not isinstance(soffset.type, ir.IntegerType) or soffset.type.width != 32:
                op = std_arith.IndexCastOp(T.i32(), soffset)
                soffset = _unwrap_value(op.result)
    aux_flags = _create_i32_constant(cache_modifier)

    rocdl.RawPtrBufferStoreOp(
        data, rsrc, offset, soffset, aux_flags
    )
