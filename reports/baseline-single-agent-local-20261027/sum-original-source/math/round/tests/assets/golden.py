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
"""Round multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/arch35 有 arch35 实现 |
| geir   | ✅   | op_graph/round_proto.h 有 REG_OP(Round) |
| aclnn  | ✅   | op_api/aclnn_round.cpp 暴露 aclnnRound / aclnnRoundDecimals 符号 |
| e2e    | ✅   | torch_npu 中 torch.round 绑定到 aclnnRound |
"""

import numpy as np
import torch

__spec__ = {
    "round": "RoundKernelSpec",
    "aclnnRound": "RoundAclnnSpec",
    "aclnnRoundDecimals": "RoundDecimalsAclnnSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnRound": "aclnn_round_golden",
        "aclnnRoundDecimals": "aclnn_round_decimals_golden",
    },
    "kernel": {"round": "round_golden"},
    "e2e": {"aclnnRound": "aclnn_round_golden"},
}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "float64": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def round_golden(x, decimals: int = 0, **kwargs):
    """Kernel golden for round. All input Tensors are numpy.ndarray."""
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    x_dtype = x.dtype

    if "int" in str(x_dtype):
        return np.round(x, decimals)

    if ori_dtype and (
        "float16" in str(ori_dtype).lower() or "bfloat16" in str(ori_dtype).lower()
    ):
        x_float = x.astype(np.float32)
        res = torch.round(torch.from_numpy(x_float), decimals=decimals).numpy()
        return res.astype(x_dtype, copy=False)
    else:
        return torch.round(torch.from_numpy(x), decimals=decimals).numpy()


def aclnn_round_golden(self, out=None, **kwargs):
    """Aclnn golden for aclnnRound. All input Tensors are torch.Tensor."""
    return [torch.round(self)]


def aclnn_round_decimals_golden(self, decimals=0, out=None, **kwargs):
    """Aclnn golden for aclnnRoundDecimals. All input Tensors are torch.Tensor."""
    return [torch.round(self, decimals=decimals)]


class _RoundCompose:
    """Third-party reference executed on the remote GPU server."""

    def __call__(self, self_=None, decimals=0, *args, **kwargs):
        del args, kwargs
        return [torch.round(self_, decimals=decimals)]


class RoundKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, decimals=0, **kwargs):
        return [round_golden(x, decimals, **kwargs)]

    third_party = {"torch": _RoundCompose}
    tolerance = _KERNEL_TOLERANCE


class RoundAclnnSpec:
    """aclnnRound spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(self, out=None, **kwargs):
        del out
        return [torch.round(self)]

    third_party = {"torch": _RoundCompose}
    tolerance = _KERNEL_TOLERANCE


class RoundDecimalsAclnnSpec:
    """aclnnRoundDecimals spec. The golden entry receives torch tensors."""

    @staticmethod
    def golden(self, decimals=0, out=None, **kwargs):
        del out
        return [torch.round(self, decimals=decimals)]

    third_party = {"torch": _RoundCompose}
    tolerance = _KERNEL_TOLERANCE
