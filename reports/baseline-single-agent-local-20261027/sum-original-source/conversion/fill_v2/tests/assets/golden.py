#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

__spec__ = {
    "fill_v2": "FILLV2KernelSpec",
}

import numpy as np
import torch


TENSOR_DTYPE_TO_TORCH_DTYPE = {
    "float64": torch.float64,
    "double": torch.float64,
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
}


TENSOR_DTYPE_TO_NUMPY_DTYPE = {
    "float64": np.float64,
    "double": np.float64,
    "float32": np.float32,
    "float": np.float32,
    "float16": np.float16,
    "half": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
}

_BINARY_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "float64": {"standard": "binary_equal"},
    "double": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
}


class ThirdPartyImpl:
    def __call__(self, dims, value=0.0, device=None, **kwargs):
        value = np.float32(value)
        out_shape = [int(d) for d in dims.tolist()]
        output_dtype = kwargs.get("output_dtypes", [None])[0]
        if output_dtype is None:
            output_dtype = "float32"
        torch_dtype = TENSOR_DTYPE_TO_TORCH_DTYPE.get(output_dtype, torch.float32)
        return [torch.full(out_shape, value, dtype=torch_dtype, device=device)]


class FILLV2KernelSpec:
    def golden(dims, value: float, **kwargs):
        """
        Kernel golden for fill_v2.
        All the parameters follow @fill_v2_def.cpp without outputs.
        All the input Tensors are numpy.ndarray.
        kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
            input_formats, output_formats, input_ori_formats, output_ori_formats,
            input_dtypes, output_dtypes.
        """
        value = np.float32(value)
        output_dtype = kwargs.get("output_dtypes", [None])[0]
        if output_dtype is None:
            output_dtype = "float32"
        torch_dtype = TENSOR_DTYPE_TO_TORCH_DTYPE.get(output_dtype, torch.float32)

        out_shape = tuple(int(d) for d in np.array(dims).flatten())

        return torch.full(out_shape, value, dtype=torch_dtype).numpy()

    third_party = {"torch": ThirdPartyImpl}

    tolerance = _BINARY_TOLERANCE
