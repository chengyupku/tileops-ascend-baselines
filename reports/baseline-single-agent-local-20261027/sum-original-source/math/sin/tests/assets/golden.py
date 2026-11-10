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
"""Sin multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/arch35 有 arch35 实现 |
| geir   | ✅   | op_graph/sin_proto.h 有 REG_OP(Sin) |
| aclnn  | ✅   | op_api/aclnn_sin.cpp 暴露 aclnnSin / aclnnInplaceSin 符号 |
| e2e    | ✅   | torch_npu 中 torch.sin 绑定到 aclnnSin |
"""

import numpy as np
import torch

__spec__ = {
    "sin": "SinKernelSpec",
    "aclnnSin": "SinAclnnSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnSin": "aclnn_sin_golden",
    },
    "kernel": {"sin": "sin_golden"},
    "e2e": {"aclnnSin": "aclnn_sin_golden"},
}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def sin_golden(x, **kwargs):
    """Kernel golden for sin. All input Tensors are numpy.ndarray."""
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    x_dtype = x.dtype

    if ori_dtype and (
        "bfloat16" in str(ori_dtype).lower() or "float16" in str(ori_dtype).lower()
    ):
        x_tensor = torch.from_numpy(x.astype(np.float32))
        output = torch.sin(x_tensor)
        return output.numpy().astype(x_dtype, copy=False)
    else:
        return np.sin(x)


def aclnn_sin_golden(self, out=None, **kwargs):
    """Aclnn golden for aclnnSin. All input Tensors are torch.Tensor."""
    return [torch.sin(self)]


class _SinCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, self_=None, *args, **kwargs):
        del args, kwargs
        return [torch.sin(self_)]


class SinKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, **kwargs):
        return [sin_golden(x, **kwargs)]

    third_party = {"torch": _SinCompose}
    tolerance = _KERNEL_TOLERANCE


class SinAclnnSpec:
    """aclnnSin spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(self, out=None, **kwargs):
        del out
        return [torch.sin(self)]

    third_party = {"torch": _SinCompose}
    tolerance = _KERNEL_TOLERANCE
