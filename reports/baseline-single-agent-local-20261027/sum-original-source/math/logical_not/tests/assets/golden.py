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
"""LogicalNot multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel 有实现 |
| geir   | ✅   | op_graph/logical_not_proto.h 有 REG_OP(LogicalNot) |
| aclnn  | ✅   | op_api/aclnn_logical_not.cpp 暴露 aclnnLogicalNot 符号 |
| e2e    | ✅   | torch_npu 中 torch.logical_not 绑定到 aclnnLogicalNot |
"""

import numpy as np
import torch

__spec__ = {
    "logical_not": "LogicalNotKernelSpec",
    "aclnnLogicalNot": "LogicalNotAclnnSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnLogicalNot": "aclnn_logical_not_golden",
    },
    "kernel": {"logical_not": "logical_not_golden"},
    "e2e": {"aclnnLogicalNot": "aclnn_logical_not_golden"},
}

_KERNEL_TOLERANCE = {
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


def logical_not_golden(x, **kwargs):
    """Kernel golden for logical_not. All input Tensors are numpy.ndarray."""
    return np.logical_not(x)


def aclnn_logical_not_golden(self, out=None, **kwargs):
    """Aclnn golden for aclnnLogicalNot. All input Tensors are torch.Tensor."""
    return [torch.logical_not(self)]


class _LogicalNotCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, self_=None, *args, **kwargs):
        del args, kwargs
        return [torch.logical_not(self_)]


class LogicalNotKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, **kwargs):
        return [logical_not_golden(x, **kwargs)]

    third_party = {"torch": _LogicalNotCompose}
    tolerance = _KERNEL_TOLERANCE


class LogicalNotAclnnSpec:
    """aclnnLogicalNot spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(self, out=None, **kwargs):
        del out
        return [torch.logical_not(self)]

    third_party = {"torch": _LogicalNotCompose}
    tolerance = _KERNEL_TOLERANCE
