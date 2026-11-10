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


def _tf_pad_compose(arr, pad_width, mode):
    """tf 原语手写 edge(=replicate)/circular(=wrap) 任意维 pad（tf.pad 无这两种模式）。
    非 eager 进程（被其他 golden 置为图模式时）用独立 Graph 兜底。"""
    import tensorflow as tf

    def build(x):
        for i, (lp, r) in enumerate(pad_width):
            lp, r = int(lp), int(r)
            if lp == 0 and r == 0:
                continue
            n = tf.shape(x)[i]
            if mode == "edge":
                if lp > 0:
                    head = tf.gather(x, [0], axis=i)
                    mult = tf.stack([lp if j == i else 1 for j in range(x.shape.rank)])
                    x = tf.concat([tf.tile(head, mult), x], axis=i)
                if r > 0:
                    tail = tf.gather(x, [tf.shape(x)[i] - 1], axis=i)
                    mult = tf.stack([r if j == i else 1 for j in range(x.shape.rank)])
                    x = tf.concat([x, tf.tile(tail, mult)], axis=i)
            else:  # circular
                if lp > 0 or r > 0:
                    reps = tf.stack(
                        [
                            (n + lp + r + n - 1) // n if j == i else 1
                            for j in range(x.shape.rank)
                        ]
                    )
                    tiled = tf.tile(x, reps)
                    begin = tf.stack(
                        [
                            tf.math.floormod(-lp, n) if j == i else 0
                            for j in range(x.shape.rank)
                        ]
                    )
                    size = tf.stack(
                        [
                            n + lp + r if j == i else tf.shape(x)[j]
                            for j in range(x.shape.rank)
                        ]
                    )
                    x = tf.slice(tiled, begin, size)
        return x

    if tf.executing_eagerly():
        return build(tf.convert_to_tensor(arr)).numpy()
    graph = tf.Graph()
    with graph.as_default():
        out_g = build(tf.convert_to_tensor(np.asarray(arr)))
    with tf.compat.v1.Session(graph=graph) as sess:
        return np.asarray(sess.run(out_g))


def framework_pad(arr, pad_width, mode):
    """非 constant 模式分派: reflect/symmetric/circular 统一实现。
    快路径: 各维 pad < size 时 tf.pad 单次反射; 否则三角波索引 gather
    支持任意宽度多重反射(np.pad 等价语义, tf.pad 有 pad<size 限制)。
    edge 走 torch F.pad(仅尾部维), dim0 有 pad 或维数不支持时
    走 _tf_pad_compose(tf 原语拼接)。"""
    import tensorflow as tf
    import torch

    ndim = arr.ndim
    mode_l = mode.lower() if isinstance(mode, str) else mode
    if mode_l in ("reflect", "symmetric", "circular"):
        fast = all(
            (int(lp) == 0 and int(r) == 0) or max(int(lp), int(r)) < int(arr.shape[i])
            for i, (lp, r) in enumerate(pad_width)
        )
        if (
            fast and mode_l != "circular"
        ):  # torch 缺 symmetric, 快路径仅 reflect/symmetric
            paddings = [[0, 0]] * ndim
            for i, (lp, r) in enumerate(pad_width):
                paddings[i] = [int(lp), int(r)]
            tf_mode = "REFLECT" if mode_l == "reflect" else "SYMMETRIC"
            if tf.executing_eagerly():
                return tf.pad(tf.convert_to_tensor(arr), paddings, mode=tf_mode).numpy()
            graph = tf.Graph()
            with graph.as_default():
                out_g = tf.pad(tf.convert_to_tensor(arr), paddings, mode=tf_mode)
            with tf.compat.v1.Session(graph=graph) as sess:
                return np.asarray(sess.run(out_g))

        def build(x):
            for i, (lp, r) in enumerate(pad_width):
                lp, r = int(lp), int(r)
                if lp == 0 and r == 0:
                    continue
                n = tf.shape(x)[i]
                total = n + lp + r
                ks = tf.range(total)
                if mode_l == "circular" or arr.shape[i] == 1:
                    idx = tf.math.floormod(ks - lp, n)
                elif mode_l == "reflect":
                    t = tf.math.floormod(ks - lp, 2 * n - 2)
                    idx = tf.where(t < n, t, 2 * n - 2 - t)
                else:  # symmetric
                    t = tf.math.floormod(ks - lp, 2 * n)
                    idx = tf.where(t < n, t, 2 * n - 1 - t)
                x = tf.gather(x, idx, axis=i)
            return x

        if tf.executing_eagerly():
            return build(tf.convert_to_tensor(arr)).numpy()
        graph = tf.Graph()
        with graph.as_default():
            out_g = build(tf.convert_to_tensor(arr))
        with tf.compat.v1.Session(graph=graph) as sess:
            return np.asarray(sess.run(out_g))
    nz = [i for i, (lp, r) in enumerate(pad_width) if lp or r]
    if not nz:
        return arr.copy()
    if ndim not in (2, 3, 4) or nz[0] == 0:
        return _tf_pad_compose(arr, pad_width, mode_l)
    # Remaining cases pad trailing axes of a 2D/3D/4D tensor.
    torch_mode = {"edge": "replicate", "circular": "circular"}[mode_l]
    flat = [0] * (2 * (ndim - 1))
    for i in range(1, ndim):
        flat[2 * (ndim - 1 - i)] = int(pad_width[i][0])
        flat[2 * (ndim - 1 - i) + 1] = int(pad_width[i][1])
    return torch.nn.functional.pad(
        torch.from_numpy(np.ascontiguousarray(arr)), flat, mode=torch_mode
    ).numpy()
