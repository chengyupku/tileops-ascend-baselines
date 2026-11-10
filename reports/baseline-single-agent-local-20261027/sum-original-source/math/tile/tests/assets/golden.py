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

import numpy as np


__golden__ = {
    "aclnn": {
        "aclnnRepeat": "aclnn_repeat_golden",
    },
    "kernel": {"tile": "tile_golden", "tile_d": "tile_golden"},
}

__spec__ = {"tile": "TileTestSpec", "tile_d": "TileDTestSpec"}


def tile_golden(x, multiples, **kwargs):
    """
    Kernel golden for tile / tile_d.
    All the parameters follow @tile_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    multiples_arr = np.array(multiples).astype(np.int64)
    multiples_val = multiples_arr.tolist()
    if isinstance(multiples_val, int):
        multiples_val = (multiples_val,)
    return np.tile(x, multiples_val)


def aclnn_repeat_golden(self, repeats=0, out=None, **kwargs):
    if hasattr(repeats, "tolist"):
        repeats = repeats.tolist()
    elif isinstance(repeats, int):
        repeats = [repeats]
    return [self.repeat(*repeats)]


class TileImpl:
    def __call__(self, x, multiples):
        import torch

        if isinstance(multiples, (np.ndarray, torch.Tensor)):
            multiples = multiples.tolist()
        elif isinstance(multiples, int):
            multiples = [multiples]
        return torch.tile(x, tuple(multiples)).contiguous()


class TileTestSpec:
    def golden(x, multiples, **kwargs):
        return [tile_golden(x, multiples, **kwargs)]

    third_party = {"torch": TileImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }


class TileDTestSpec:
    def golden(x, multiples, **kwargs):
        return [tile_golden(x, multiples, **kwargs)]

    third_party = {"torch": TileImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
