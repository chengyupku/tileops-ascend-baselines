#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Cast multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/cast_apt.cpp 有 arch35 实现 |
| geir   | ✅   | op_graph/cast_proto.h 有 REG_OP(Cast) |
| aclnn  | ✅   | op_api/aclnn_cast.cpp 暴露 aclnnCast 符号 |
| e2e    | ✅   | torch_npu 中 torch.Tensor.to / torch.cast 绑定到 aclnnCast |

kernel 与 geir 共用一个注册键（算子蛇形名），geir 不另写 Spec 类。
"""

import numpy as np

__spec__ = {
    "cast": "CastKernelSpec",
    "aclnnCast": "CastAclnnSpec",
    "torch.Tensor.to": "CastE2eSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnCast": "aclnn_cast_golden",
    },
    "kernel": {"cast": "cast_golden"},
    "e2e": {"aclnnCast": "aclnn_cast_golden"},
}

_DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    16: "complex64",
    17: "complex128",
    27: "bfloat16",
    29: "int4",
    30: "uint1",
    33: "complex32",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    40: "float4_e2m1",
    41: "float4_e1m2",
}

_SPECIAL_DTYPES = (
    "bfloat16",
    "int4",
    "float8_e5m2",
    "float8_e4m3fn",
    "float4_e2m1",
    "float4_e1m2",
    "hifloat8",
)

# 判据声明: 覆盖 cast_def.cpp 注册的全部 dtype。
# 浮点一律 cross_check —— 只有 cross_check 才会让 TTK 取三方(GPU)输出并开 golden_mode=Promote;
# 整型/布尔位精确，用 binary_equal; 复数走 stat_rel_err。
_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "float64": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
    "bool": {"standard": "binary_equal"},
    "complex64": {"standard": "stat_rel_err"},
    "complex128": {"standard": "stat_rel_err"},
}


def _resolve_custom_numpy_dtype(dtype_str):
    if dtype_str == "bfloat16":
        from ml_dtypes import bfloat16

        return bfloat16
    elif dtype_str == "int4":
        from ml_dtypes import int4

        return int4
    elif dtype_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    elif dtype_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    elif dtype_str == "hifloat8":
        from en_dtypes import hifloat8

        return hifloat8
    elif dtype_str == "float4_e2m1":
        from ml_dtypes import float4_e2m1

        return float4_e2m1
    elif dtype_str == "float4_e1m2":
        from ml_dtypes import float4_e1m2

        return float4_e1m2
    return None


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def cast_golden(x, dst_type: int, **kwargs):
    """
    Kernel golden for cast.
    All the parameters follow @cast_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
              input_formats, output_formats, input_ori_formats, output_ori_formats,
              input_dtypes, output_dtypes.
    """
    dst_type_str = _DATA_TYPE_INT_TO_STR.get(dst_type, str(dst_type))
    if (x.dtype.name == "bfloat16" and dst_type_str == "hifloat8") or (
        x.dtype.name == "hifloat8" and dst_type_str == "bfloat16"
    ):
        np_dtype = _resolve_custom_numpy_dtype(dst_type_str)
        return x.astype(np.float32).astype(np_dtype)
    elif dst_type_str in _SPECIAL_DTYPES:
        np_dtype = _resolve_custom_numpy_dtype(dst_type_str)
        return x.astype(np_dtype)
    elif dst_type_str == "complex32":
        _shape = list(x.shape)
        x = x.reshape(_shape + [1])
        imag = np.zeros(_shape + [1], dtype=np.float16)
        res = np.concatenate((x, imag), axis=-1)
        return res
    elif dst_type_str == "bool":
        return x.astype(np.bool_)
    else:
        return x.astype(getattr(np, dst_type_str))


def aclnn_cast_golden(self, dtype=0, out=None, **kwargs):
    """
    Aclnn golden for aclnnCast.
    Parameters follow @aclnnCastGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    from ttk.utilities import acl_to_torch_dtype

    torch_dtype = acl_to_torch_dtype([dtype])[0]
    return self.to(dtype=torch_dtype)


def _resolve_torch_dtype(dtype_int):
    """Map aclnn dtype int to torch dtype, falling back to ttk utilities."""
    from ttk.utilities import acl_to_torch_dtype

    return acl_to_torch_dtype([dtype_int])[0]


class _CastCompose:
    """Third-party reference executed on the remote GPU server.

    torch.Tensor.to is the competitor interface; the kernel golden uses
    numpy.astype, so the two paths are independent implementations of the
    cast semantics and can cross-validate each other.
    """

    def __call__(self, self_=None, dtype=0, *args, **kwargs):
        del args, kwargs
        torch_dtype = _resolve_torch_dtype(dtype)
        return [self_.to(dtype=torch_dtype)]


class CastKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, dst_type, **kwargs):
        result = cast_golden(x, dst_type, **kwargs)
        output_dtype = _output_dtype(kwargs, 0, str(result.dtype))
        np_dtype = _resolve_custom_numpy_dtype(output_dtype)
        if np_dtype is not None:
            return [result.astype(np_dtype, copy=False)]
        target = getattr(np, output_dtype, None)
        if target is not None:
            return [result.astype(target, copy=False)]
        return [result]

    third_party = {"torch": _CastCompose}
    tolerance = _KERNEL_TOLERANCE


class CastAclnnSpec:
    """aclnnCast spec. The golden entry receives torch tensors.

    The parameter name follows aclnn_cast.h, where the input is named self.
    """

    @staticmethod
    def golden(self, dtype=0, out=None, **kwargs):
        del out
        torch_dtype = _resolve_torch_dtype(dtype)
        return [self.to(dtype=torch_dtype)]

    third_party = {"torch": _CastCompose}
    tolerance = _KERNEL_TOLERANCE


class _CastE2eCompose:
    """Third-party reference for the E2E path (api ``torch.Tensor.to``).

    torch.Tensor.to is a C builtin without an inspectable signature, so the
    server-side api mode fails with "no signature found for builtin". This
    compose re-declares the call in plain Python; tensors are recognized by
    instance type and the target dtype arrives under the 'dtype' attribute
    key (string form, e.g. "torch.float32").
    """

    def __call__(self, *args, **kwargs):
        import torch

        tensor = None
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                tensor = v
                break
        if tensor is None:
            for v in args:
                if isinstance(v, torch.Tensor):
                    tensor = v
                    break
        dtype = kwargs.get("dtype")
        if dtype is None:
            for v in args:
                if not isinstance(v, torch.Tensor) and not isinstance(v, (list, tuple)):
                    dtype = v
                    break
        if tensor is None or dtype is None:
            raise ValueError(
                "cast e2e compose expects one input tensor and a 'dtype' attribute"
            )
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.rsplit(".", 1)[-1])
        return [tensor.to(dtype=dtype)]


class CastE2eSpec:
    """E2E spec, keyed by the dotted api name ``torch.Tensor.to`` in __spec__.

    third_party is a compose class so the XPU dispatch runs in spec mode
    (server instantiates and binds params by name) instead of api mode.
    """

    third_party = {"torch": _CastE2eCompose}
