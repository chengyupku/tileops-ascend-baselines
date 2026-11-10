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

import os

__spec__ = {
    "parallel_concat": "ParallelConcatTestSpec",  # kernel/GEIR CSV op_name (snake_case)
    "ParallelConcat": "ParallelConcatTestSpec",  # kernel/GEIR CSV op_name (官方 IR CamelCase)
    "parallelconcat": "ParallelConcatTestSpec",  # 小写无下划线拼写兜底
    "tf.raw_ops.ParallelConcat": "ParallelConcatTestSpec",  # E2E CSV api_name
}

_TF_PARALLEL_CONCAT = None


def _tf_parallel_concat():
    """构建并缓存 tf.function 包装的 tf.raw_ops.ParallelConcat（进程内单例）。"""
    global _TF_PARALLEL_CONCAT
    if _TF_PARALLEL_CONCAT is None:
        os.environ.setdefault(
            "TF_CPP_MIN_LOG_LEVEL", "3"
        )  # 抑制 TF INFO/WARNING 污染 TTK 日志
        import tensorflow as tf

        @tf.function(autograph=False)
        def _parallel_concat(values, shape):
            return tf.raw_ops.ParallelConcat(values=values, shape=shape)

        _TF_PARALLEL_CONCAT = _parallel_concat
    return _TF_PARALLEL_CONCAT


def _flatten_inputs(values):
    """把动态输入 values 归一化为张量列表，兼容各通路到达形态：

    - kernel/GEIR：golden(values=(a0, ..., aN-1))——values 为 DYNAMIC_INPUT N 张量
      折叠后的 tuple/list（嵌套）；
    - E2E：golden(values=[a0, ..., aN-1])——TensorList 由 build_args 组装为 list；
    - 单张量：golden(values=a0)——N=1 或非折叠形态；
    - 双层包裹：values=((a0, ...),)——整体作单参到达时解一层。
    值内混入的 dict/标量（冗余元信息）被过滤；未知类型立刻报错（不静默丢弃）。
    """
    import numpy as np
    import tensorflow as tf
    import torch

    if isinstance(values, (list, tuple)):
        items = list(values)
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            items = list(items[0])
    elif values is None:
        items = []
    else:
        items = [values]

    tensors = []
    for v in items:
        if isinstance(v, (torch.Tensor, np.ndarray, tf.Tensor)):
            tensors.append(v)
        elif isinstance(v, (dict, int, float, bool, type(None))):
            continue  # 冗余元信息（attrs/rank/ws 等），不参与求值
        else:
            raise ValueError(
                f"parallel_concat golden: unexpected input item of type {type(v)!r}"
            )
    return tensors


def _to_tf(x):
    """numpy.ndarray / torch.Tensor -> tf.Tensor（逐比特）。

    bfloat16 一律经 uint16 位视图 + tf.bitcast 转 tf.bfloat16：不依赖 TF 是否原生
    识别 ml_dtypes.bfloat16，仅靠「16 位无符号搬运 + 按位重解释」即逐位保真，跨 TF
    版本 / 跨 numpy-bf16 表示均成立（torch/numpy 两条输入路统一走此桥）。
    """
    import numpy as np
    import tensorflow as tf
    import torch

    if isinstance(x, tf.Tensor):
        return x
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            # torch 无 bf16→numpy 直转：先取 uint16 位视图，再 bitcast 回 bf16
            u16 = t.view(torch.int16).numpy().view(np.uint16)
            return tf.bitcast(tf.constant(u16), tf.bfloat16)
        return tf.constant(t.numpy())
    arr = np.ascontiguousarray(x)
    if arr.dtype.name == "bfloat16":  # ml_dtypes.bfloat16
        return tf.bitcast(tf.constant(arr.view(np.uint16)), tf.bfloat16)
    return tf.constant(arr)


def _from_tf(out, like):
    """tf.Tensor -> 与 like 同形态的输出（torch.Tensor / numpy.ndarray；逐比特，
    bfloat16 经 int16 位视图还原）。"""
    import numpy as np
    import tensorflow as tf
    import torch

    if isinstance(like, tf.Tensor):
        return out
    if isinstance(like, torch.Tensor):
        arr = out.numpy()
        if like.dtype == torch.bfloat16:
            # tf.bfloat16 -> ml_dtypes bfloat16 -> int16 位视图 -> torch.bfloat16
            return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(
                torch.bfloat16
            )
        return torch.from_numpy(arr)
    return out.numpy()


class ParallelConcatTestSpec:
    """parallel_concat 测试规范。

    golden：kernel/GEIR 流程收 numpy.ndarray（values 动态输入为 tuple/list 嵌套），
    torch 输入（E2E 类通路）逐比特转入 tf 计算后按原形态返回。返回 [output_data]。
    属性 shape/n/N 被接收但不参与求值（输出 shape 由输入完全确定：
    (N,) + values.shape[1:]，属性为冗余校验约束；shape 缺席时按该契约推导后作为
    竞品 API tf.raw_ops.ParallelConcat 的必选 attr 传入）。
    """

    def golden(values=None, shape=None, n=None, N=None, **kwargs):
        # 各通路到达约定：
        #   kernel/GEIR：golden(values, shape=..., n=..., N=...)——values 为动态输入
        #     N 张量的 tuple/list 嵌套（DYNAMIC_INPUT 折叠为首个位置参），shape/n/N 为
        #     关键字属性；输入/输出为 numpy.ndarray。
        #   E2E（tf.raw_ops.ParallelConcat）：golden(values, shape)——shape 作第二位置参
        #     由 ParamPlan.build_args 依 API overload 填充；输入/输出为 torch.Tensor。
        # shape/n/N 仅作冗余校验约束，不参与求值：输出形态由输入完全确定
        # ((N,) + values.shape[1:])，故一律按契约由输入重推，忽略传入的冗余 shape。
        items = _flatten_inputs(values)
        if not items:
            raise ValueError("parallel_concat golden: no input tensors received")

        tensors = [_to_tf(v) for v in items]
        # 冗余属性无论缺席与否，均按算子契约由输入重推 shape，杜绝 E2E 侧 shape 张量
        # 与 attr 混入造成的错绑：shape == (N,) + values.shape[1:]
        out_shape = [len(tensors)] + [int(d) for d in tensors[0].shape[1:]]
        out = _tf_parallel_concat()(tensors, out_shape)
        return [_from_tf(out, items[0])]

    class TfParallelConcatImpl:
        """tf provider 竞品实现（spec reference_oracle: tensorflow.raw_ops.ParallelConcat）。"""

        def __init__(self, *, shape=None, n=None, N=None, **kwargs):
            self.shape = shape
            self.n = n if n is not None else N

        def __call__(self, *, values=None, **kwargs):
            if values is None:
                raise ValueError("TfParallelConcatImpl: no 'values' input received")
            tensors = list(values) if isinstance(values, (list, tuple)) else [values]
            shape = self.shape
            if shape is None:
                # 冗余属性缺席时按算子契约推导：shape == (N,) + values.shape[1:]
                shape = [len(tensors)] + [int(d) for d in tensors[0].shape[1:]]
            shape = [int(d) for d in shape]
            import tensorflow as tf

            return [tf.raw_ops.ParallelConcat(values=tensors, shape=shape)]

    third_party = {"tf": TfParallelConcatImpl}

    # 纯数据搬运 → binary_equal（spec numerical_tolerance 全 dtype bitwise_equal）；
    # cross_check override（spec cross_check.level=L1）作用于浮点 dtype，需上方 third_party
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "uint64": {"standard": "binary_equal"},
        "bool": {"standard": "binary_equal"},
    }
