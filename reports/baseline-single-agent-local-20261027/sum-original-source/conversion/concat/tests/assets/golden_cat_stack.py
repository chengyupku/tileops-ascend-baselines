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
import torch

__spec__ = {"aclnnCat": "CatStackSpec", "aclnnStack": "CatStackSpec"}


class CatStackSpec:
    """golden for aclnnCat / aclnnStack. Input tensors may be non-contiguous
    strided views; torch CPU handles them natively. Pure data-movement ops
    require bit-exact results."""

    tolerance = {
        d: {"standard": "binary_equal"}
        for d in (
            "float32",
            "float64",
            "float16",
            "bfloat16",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "bool",
        )
    }

    third_party = {"torch": "torch.cat"}

    @staticmethod
    def golden(tensors, dim=0, out=None, **kwargs):
        tensors = list(tensors)
        # cat/stack 派发判据, 优先级从高到低:
        # 1) out 秩判据(确定性): stack 输出恒比输入多一轴, cat 与输入同秩;
        #    TTK 会把 API 签名中的 out 以 torch.Tensor(含输出 shape) 传入
        # 2) dim 合法域判据: cat ∈ [-R, R-1], stack ∈ [-R-1, R] (R=输入秩);
        #    dim 落在 cat 域外(==R 或 ==-(R+1)) 时必为 stack
        # 3) 用例名启发(兜底, 仅前两者不可用时)
        in_rank = tensors[0].dim() if tensors else 0
        if out is not None and hasattr(out, "ndim"):
            is_stack = out.ndim == in_rank + 1
        else:
            if dim == in_rank or dim == -(in_rank + 1):
                is_stack = True
            else:
                name = kwargs.get("testcase_name", "")
                is_stack = (
                    "stack" in name
                    or "_st_" in name
                    or "stneg" in name
                    or "stnum" in name
                    or name.startswith("st_")
                )
        fn = torch.stack if is_stack else torch.cat
        return [fn(tensors, dim=dim)]
