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
"""Fill multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel 有实现 |
| geir   | ✅   | op_graph/fill_proto.h 有 REG_OP(Fill) |
| aclnn  | ✅   | op_api/aclnn_fill_scalar.cpp / aclnn_fill_tensor.cpp 暴露 aclnnInplaceFillScalar / aclnnInplaceFillTensor |
| e2e    | ✅   | torch_npu 中 torch.fill_ / torch.Tensor.fill_ 绑定到 aclnnInplaceFillScalar |
"""

import numpy as np

__spec__ = {
    "fill": "FillKernelSpec",
    "aclnnInplaceFillScalar": "FillScalarAclnnSpec",
    "aclnnInplaceFillTensor": "FillTensorAclnnSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnInplaceFillScalar": "aclnn_inplace_fill_scalar_golden",
        "aclnnInplaceFillTensor": "aclnn_inplace_fill_tensor_golden",
    },
    "kernel": {"fill": "fill_golden"},
    "e2e": {"aclnnInplaceFillScalar": "aclnn_inplace_fill_scalar_golden"},
}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "bool": {"standard": "binary_equal"},
}


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def fill_golden(dims, value, **kwargs):
    """Kernel golden for fill. All input Tensors are numpy.ndarray."""
    shape = tuple(dims.tolist())
    return np.full(shape, value[0], dtype=value.dtype)


def aclnn_inplace_fill_scalar_golden(selfRef, value, **kwargs):
    """Aclnn golden for aclnnInplaceFillScalar. All input Tensors are torch.Tensor.

    Parameters follow @aclnnInplaceFillScalarGetWorkspaceSize without workspaceSize & executor.
    """
    result = selfRef.clone()
    result.fill_(value)
    return [result]


def aclnn_inplace_fill_tensor_golden(selfRef, value, **kwargs):
    """Aclnn golden for aclnnInplaceFillTensor. All input Tensors are torch.Tensor.

    Parameters follow @aclnnInplaceFillTensorGetWorkspaceSize without workspaceSize & executor.
    """
    result = selfRef.clone()
    result.fill_(value.item())
    return [result]


class _FillScalarCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, selfRef, value, *args, **kwargs):
        del args, kwargs
        result = selfRef.clone()
        result.fill_(value)
        return [result]


class _FillTensorCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, selfRef, value, *args, **kwargs):
        del args, kwargs
        result = selfRef.clone()
        result.fill_(value.item())
        return [result]


class FillKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(dims, value, **kwargs):
        return [fill_golden(dims, value, **kwargs)]

    third_party = {"torch": _FillScalarCompose}
    tolerance = _KERNEL_TOLERANCE


class FillScalarAclnnSpec:
    """aclnnInplaceFillScalar spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(selfRef, value, **kwargs):
        result = selfRef.clone()
        result.fill_(value)
        return [result]

    third_party = {"torch": _FillScalarCompose}
    tolerance = _KERNEL_TOLERANCE


class FillTensorAclnnSpec:
    """aclnnInplaceFillTensor spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(selfRef, value, **kwargs):
        result = selfRef.clone()
        result.fill_(value.item())
        return [result]

    third_party = {"torch": _FillTensorCompose}
    tolerance = _KERNEL_TOLERANCE
