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
"""IsFinite multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel 有实现 |
| geir   | ✅   | op_graph/is_finite_proto.h 有 REG_OP(IsFinite) |
| aclnn  | ✅   | op_api/aclnn_is_finite.cpp 暴露 aclnnIsFinite 符号 |
| e2e    | ✅   | torch_npu 中 torch.isfinite 绑定到 aclnnIsFinite |
"""

import numpy as np
import torch

__spec__ = {
    "is_finite": "IsFiniteKernelSpec",
    "aclnnIsFinite": "IsFiniteAclnnSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnIsFinite": "aclnn_is_finite_golden",
    },
    "kernel": {"is_finite": "is_finite_golden"},
    "e2e": {"aclnnIsFinite": "aclnn_is_finite_golden"},
}

_KERNEL_TOLERANCE = {
    "bool": {"standard": "binary_equal"},
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
}


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def is_finite_golden(x, **kwargs):
    """Kernel golden for is_finite. All input Tensors are numpy.ndarray."""
    return np.isfinite(x)


def aclnn_is_finite_golden(self, out=None, **kwargs):
    """Aclnn golden for aclnnIsFinite. All input Tensors are torch.Tensor."""
    return [torch.isfinite(self)]


class _IsFiniteCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, self_=None, *args, **kwargs):
        del args, kwargs
        return [torch.isfinite(self_)]


class IsFiniteKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, **kwargs):
        return [is_finite_golden(x, **kwargs)]

    third_party = {"torch": _IsFiniteCompose}
    tolerance = _KERNEL_TOLERANCE


class IsFiniteAclnnSpec:
    """aclnnIsFinite spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(self, out=None, **kwargs):
        del out
        return [torch.isfinite(self)]

    third_party = {"torch": _IsFiniteCompose}
    tolerance = _KERNEL_TOLERANCE
