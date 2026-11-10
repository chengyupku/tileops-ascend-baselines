#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
prod_env_mat_a Golden TestSpec

Golden: TensorFlow implementation (SE §7, DeepMD-kit compatible)
ThirdParty: pure torch implementation for GEIR cross_check

Golden 计算逻辑与 SE 文档第7章一致，全程使用 TensorFlow API 做数学计算；
ThirdParty 为纯 torch 独立实现相同数学逻辑（不共享 Golden 的 TF 前处理路径，不依赖 TF）。
"""

import tensorflow as tf
import torch

__spec__ = {
    "prod_env_mat_a": "ProdEnvMatATestSpec",
}


# ---------------------------------------------------------------------------
#  Shared math: spline5_switch coefficient extraction
# ---------------------------------------------------------------------------


def _spline5_switch(nr_val, rmin, rmax):
    """五次样条平滑函数 (与 SE §1.1 / §7 一致)。

    返回 (sw, dsw) Python float，供循环中使用。
    计算 sw/dsw 的公式用 TF eager tensor 完成，最终提取标量。
    """
    if nr_val < rmin:
        return 1.0, 0.0
    elif nr_val < rmax:
        uu = (nr_val - rmin) / (rmax - rmin)
        du = 1.0 / (rmax - rmin)
        sw = uu * uu * uu * (-6.0 * uu * uu + 15.0 * uu - 10.0) + 1.0
        dsw = (
            3.0 * uu * uu * (-6.0 * uu * uu + 15.0 * uu - 10.0)
            + uu * uu * uu * (-12.0 * uu + 15.0)
        ) * du
        return float(sw), float(dsw)
    else:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
#  Golden — TensorFlow implementation (SE §7)
# ---------------------------------------------------------------------------


class ProdEnvMatATestSpec:
    """One TestSpec shared by Kernel and GEIR.

    Golden receives tf.Tensor (or array-like), returns list[tf.Tensor].
        ThirdParty receives array-like (numpy/tf/torch/list), returns list[torch.Tensor].
    """

    def golden(
        coord,
        type,
        natoms,
        box,
        mesh,
        davg,
        dstd,
        *,
        rcut_a,
        rcut_r,
        rcut_r_smth,
        sel_a,
        sel_r,
        **kwargs,
    ):
        """Golden 使用 TensorFlow 实现 (SE §7)。

        所有数学计算 (rsqrt, sqrt, reduce_sum, 加减乘除) 均使用 TF API。
        输出缓冲区使用 tf.Variable 存储（支持索引赋值），返回时转为 tf.Tensor
        以兼容 TTK 比较框架（registry.py 调用 golden.reshape([-1])，tf.Variable 无 reshape 方法）。
        """
        # ---- 解析维度 ----
        nsample = int(coord.shape[0])
        nall = int(coord.shape[1]) // 3
        nloc = int(natoms[0])
        nnei = int(sum(sel_a))
        ndescrpt = nnei * 4

        # ---- sec: 每种类型的邻居起始偏移 ----
        sec = [0]
        for s in sel_a:
            sec.append(sec[-1] + int(s))

        # ---- 计算精度: 跟随 coord dtype (Enable=float32, Promote=float64) ----
        # SE §5.5 要求全程 float32 计算; Promote 模式下 cross_check 需要高精度基准,
        # 因此 Golden 根据输入 dtype 选择计算精度, 不强制降回 float32。
        coord_t = tf.convert_to_tensor(coord)
        tf_dtype = tf.float64 if coord_t.dtype == tf.float64 else tf.float32
        if tf_dtype != coord_t.dtype:
            coord_t = tf.cast(coord_t, tf_dtype)

        # ---- 转换为 TF Tensor 做计算 ----
        type_arr = tf.convert_to_tensor(type, dtype=tf.int32)
        tf.convert_to_tensor(natoms, dtype=tf.int32)
        davg_flat = tf.reshape(tf.convert_to_tensor(davg, dtype=tf_dtype), [-1])
        dstd_flat = tf.reshape(tf.convert_to_tensor(dstd, dtype=tf_dtype), [-1])

        # ---- 输出缓冲区 (tf.Variable 存储，TF 计算) ----
        # float 输出跟随计算 dtype; nlist 固定 int32
        descrpt = tf.Variable(tf.zeros((nsample, nloc * nnei * 4), dtype=tf_dtype))
        descrpt_deriv = tf.Variable(
            tf.zeros((nsample, nloc * nnei * 12), dtype=tf_dtype)
        )
        rij_out = tf.Variable(tf.zeros((nsample, nloc * nnei * 3), dtype=tf_dtype))
        nlist_out = tf.Variable(tf.cast(tf.fill((nsample, nloc * nnei), -1), tf.int32))

        # ---- 提前将 coord 转为 TF Tensor 用于循环索引 ----
        coord_3d = tf.reshape(coord_t, [nsample, nall, 3])

        # ---- ntypes: davg/dstd 的第一维大小，用于 atom_type 越界保护 ----
        ntypes = (
            int(davg.shape[0]) if hasattr(davg, "shape") else int(tf.shape(davg)[0])
        )

        for frame in range(nsample):
            coord_frame = coord_3d[frame]  # (nall, 3) TF tensor
            type_frame = type_arr[frame]  # (nall,)   TF tensor

            for atom_idx in range(nloc):
                atom_type_early = int(type_frame[atom_idx])
                # SE §7: type < 0 表示虚拟原子，跳过;
                # type >= ntypes 表示非法类型编号（davg/dstd 第一维为 ntypes），跳过以避免索引越界。
                if atom_type_early < 0 or atom_type_early >= ntypes:
                    continue

                i_coord = coord_frame[atom_idx]  # (3,)

                # ---- 阶段1: 查找并排序邻居 ----
                # @constraint: nr <= rcut_r (SE §5.2)
                # 距离计算用 TF API
                i_t = tf.constant(i_coord, dtype=tf_dtype)
                neighbors = []
                for j_idx in range(nall):
                    if j_idx == atom_idx:
                        continue
                    if int(type_frame[j_idx]) < 0:
                        continue
                    j_t = tf.constant(coord_frame[j_idx], dtype=tf_dtype)
                    diff = j_t - i_t
                    nr2_t = tf.reduce_sum(diff * diff)
                    nr_val = float(tf.sqrt(nr2_t))
                    if nr_val <= rcut_r:
                        neighbors.append((int(type_frame[j_idx]), nr_val, j_idx))

                # 按 (type, distance) 升序排序
                neighbors.sort(key=lambda x: (x[0], x[1]))

                # 按 sel_a 分段填充 nlist
                type_counts = [0] * len(sel_a)
                for t_val, d_val, j in neighbors:
                    if t_val < len(sel_a) and type_counts[t_val] < sel_a[t_val]:
                        out_idx = atom_idx * nnei + sec[t_val] + type_counts[t_val]
                        nlist_out[frame, out_idx].assign(j)
                        type_counts[t_val] += 1

                # ---- 阶段2: 计算环境矩阵 ----
                atom_type = int(type_frame[atom_idx])
                for ii in range(nnei):
                    idx_value = ii * 4
                    idx_deriv = ii * 12
                    nbor_idx = int(nlist_out[frame, atom_idx * nnei + ii])

                    if nbor_idx >= 0:
                        # 用 TF Tensor 计算
                        j_t = tf.constant(coord_frame[nbor_idx], dtype=tf_dtype)
                        rr = j_t - i_t  # (3,)
                        nr2 = tf.reduce_sum(rr * rr)
                        # nr2 极小值保护: 坐标极近时 nr2 趋 0 → inr=rsqrt(nr2) 极大 →
                        # inr4=inr^4 溢出 float32 为 inf → inr3=inf*nr=nan/inf → 传播到 vv0-vv11。
                        # safe_nr2=max(nr2,1e-10) 使 inr3 上界 ~1e15（float32 安全），
                        # 对正常 nr2（>>1e-10）无影响。
                        safe_nr2 = tf.maximum(nr2, tf.constant(1e-10, dtype=tf_dtype))
                        inr = tf.math.rsqrt(safe_nr2)
                        nr = safe_nr2 * inr
                        inr2 = inr * inr
                        inr4 = inr2 * inr2
                        inr3 = inr4 * nr

                        # spline5_switch (TF 计算)
                        nr_val = float(nr)
                        rmin = float(rcut_r_smth)
                        rmax = float(rcut_r)
                        sw_f, dsw_f = _spline5_switch(nr_val, rmin, rmax)
                        sw = tf.constant(sw_f, dtype=tf_dtype)
                        dsw = tf.constant(dsw_f, dtype=tf_dtype)

                        nr2_val = float(nr2)
                        rr0 = float(rr[0])
                        rr1 = float(rr[1])
                        rr2 = float(rr[2])
                        inr_f = float(inr)
                        inr2_f = float(inr2)
                        inr3_f = float(inr3)

                        # 4 分量描述子 (TF 计算)
                        # nr_val/nr2_val 除零保护: 原子坐标重合时 nr=0 导致 ZeroDivisionError
                        safe_nr_val = max(nr_val, 1e-10)
                        safe_nr2_val = max(nr2_val, 1e-10)
                        dd0 = tf.constant(1.0 / safe_nr_val, dtype=tf_dtype) * sw
                        dd1 = tf.constant(rr0 / safe_nr2_val, dtype=tf_dtype) * sw
                        dd2 = tf.constant(rr1 / safe_nr2_val, dtype=tf_dtype) * sw
                        dd3 = tf.constant(rr2 / safe_nr2_val, dtype=tf_dtype) * sw

                        # 12 分量导数 (TF 计算)
                        vv0 = tf.constant(
                            rr0 * inr3_f, dtype=tf_dtype
                        ) * sw - dd0 * dsw * tf.constant(rr0 * inr_f, dtype=tf_dtype)
                        vv1 = tf.constant(
                            rr1 * inr3_f, dtype=tf_dtype
                        ) * sw - dd0 * dsw * tf.constant(rr1 * inr_f, dtype=tf_dtype)
                        vv2 = tf.constant(
                            rr2 * inr3_f, dtype=tf_dtype
                        ) * sw - dd0 * dsw * tf.constant(rr2 * inr_f, dtype=tf_dtype)
                        vv3 = tf.constant(
                            2.0 * rr0 * rr0 * inr2_f - inr_f, dtype=tf_dtype
                        ) * sw - dd1 * dsw * tf.constant(rr0 * inr_f, dtype=tf_dtype)
                        vv4 = tf.constant(
                            2.0 * rr0 * rr1 * inr2_f, dtype=tf_dtype
                        ) * sw - dd1 * dsw * tf.constant(rr1 * inr_f, dtype=tf_dtype)
                        vv5 = tf.constant(
                            2.0 * rr0 * rr2 * inr2_f, dtype=tf_dtype
                        ) * sw - dd1 * dsw * tf.constant(rr2 * inr_f, dtype=tf_dtype)
                        vv6 = tf.constant(
                            2.0 * rr1 * rr0 * inr2_f, dtype=tf_dtype
                        ) * sw - dd2 * dsw * tf.constant(rr0 * inr_f, dtype=tf_dtype)
                        vv7 = tf.constant(
                            2.0 * rr1 * rr1 * inr2_f - inr_f, dtype=tf_dtype
                        ) * sw - dd2 * dsw * tf.constant(rr1 * inr_f, dtype=tf_dtype)
                        vv8 = tf.constant(
                            2.0 * rr1 * rr2 * inr2_f, dtype=tf_dtype
                        ) * sw - dd2 * dsw * tf.constant(rr2 * inr_f, dtype=tf_dtype)
                        vv9 = tf.constant(
                            2.0 * rr2 * rr0 * inr2_f, dtype=tf_dtype
                        ) * sw - dd3 * dsw * tf.constant(rr0 * inr_f, dtype=tf_dtype)
                        vv10 = tf.constant(
                            2.0 * rr2 * rr1 * inr2_f, dtype=tf_dtype
                        ) * sw - dd3 * dsw * tf.constant(rr1 * inr_f, dtype=tf_dtype)
                        vv11 = tf.constant(
                            2.0 * rr2 * rr2 * inr2_f - inr_f, dtype=tf_dtype
                        ) * sw - dd3 * dsw * tf.constant(rr2 * inr_f, dtype=tf_dtype)

                        vv_list = [
                            vv0,
                            vv1,
                            vv2,
                            vv3,
                            vv4,
                            vv5,
                            vv6,
                            vv7,
                            vv8,
                            vv9,
                            vv10,
                            vv11,
                        ]
                        dd_list = [dd0, dd1, dd2, dd3]

                        # 存储 rij
                        rij_out[frame, atom_idx * nnei * 3 + ii * 3 + 0].assign(rr0)
                        rij_out[frame, atom_idx * nnei * 3 + ii * 3 + 1].assign(rr1)
                        rij_out[frame, atom_idx * nnei * 3 + ii * 3 + 2].assign(rr2)

                        # 归一化并存储
                        base_davg = atom_type * ndescrpt + idx_value
                        for k in range(12):
                            descrpt_deriv[
                                frame, atom_idx * nnei * 12 + idx_deriv + k
                            ].assign(
                                float(vv_list[k])
                                / max(float(dstd_flat[base_davg + k // 3]), 1e-10)
                            )
                        for k in range(4):
                            descrpt[frame, atom_idx * nnei * 4 + idx_value + k].assign(
                                (float(dd_list[k]) - float(davg_flat[base_davg + k]))
                                / max(float(dstd_flat[base_davg + k]), 1e-10)
                            )
                    else:
                        # 无邻居: descrpt = -davg/dstd (SE §5.2)
                        base_davg = atom_type * ndescrpt + idx_value
                        for k in range(4):
                            descrpt[frame, atom_idx * nnei * 4 + idx_value + k].assign(
                                -float(davg_flat[base_davg + k])
                                / max(float(dstd_flat[base_davg + k]), 1e-10)
                            )

        # ---- 返回 numpy array（非 tf.Variable/tf.Tensor），兼容 TTK 比较框架 ----
        # TTK comparison/registry.py:58 调用 golden.reshape([-1])，
        # tf.Variable 和 EagerTensor 均无 reshape 方法，numpy array 有。
        return [
            descrpt.numpy(),
            descrpt_deriv.numpy(),
            rij_out.numpy(),
            nlist_out.numpy(),
        ]

    # ------------------------------------------------------------------
    #  ThirdParty — pure torch implementation for GEIR cross_check
    # ------------------------------------------------------------------

    class ThirdPartyImpl:
        """torch provider: 纯 torch 独立实现相同数学逻辑（不依赖 TF）。"""

        def __init__(
            self,
            *,
            rcut_a=1.0,
            rcut_r=1.0,
            rcut_r_smth=1.0,
            sel_a=None,
            sel_r=None,
            **kwargs,
        ):
            self.rcut_a = float(rcut_a)
            self.rcut_r = float(rcut_r)
            self.rcut_r_smth = float(rcut_r_smth)
            self.sel_a = list(sel_a) if sel_a else []
            self.sel_r = list(sel_r) if sel_r else []

        def __call__(self, coord, type, natoms, box, mesh, davg, dstd, **kwargs):
            # ---- 输入统一转 torch (兼容 numpy/tf tensor/列表等 array-like 输入) ----
            coord_t = torch.as_tensor(coord)
            type_t = torch.as_tensor(type, dtype=torch.int32)
            natoms_t = torch.as_tensor(natoms, dtype=torch.int32)
            davg_t = torch.as_tensor(davg)
            dstd_t = torch.as_tensor(dstd)

            # ---- 计算精度: 跟随 coord dtype (float64 用 float64, 否则 float32),
            # 与 golden 的 tf_dtype 逻辑一致 ----
            calc_dtype = (
                torch.float64 if coord_t.dtype == torch.float64 else torch.float32
            )
            if coord_t.dtype != calc_dtype:
                coord_t = coord_t.to(calc_dtype)

            # ---- 解析维度 ----
            nsample = int(coord_t.shape[0])
            nall = int(coord_t.shape[1]) // 3
            nloc = int(natoms_t[0])
            nnei = int(sum(self.sel_a))
            ndescrpt = nnei * 4

            sec = [0]
            for s in self.sel_a:
                sec.append(sec[-1] + int(s))

            # ---- 展平 davg/dstd 为一维 torch tensor (计算 dtype) ----
            davg_flat = davg_t.to(calc_dtype).reshape(-1)
            dstd_flat = dstd_t.to(calc_dtype).reshape(-1)

            # ---- 输出 torch tensor (float 输出跟随计算 dtype; nlist 固定 int32) ----
            descrpt = torch.zeros((nsample, nloc * nnei * 4), dtype=calc_dtype)
            descrpt_deriv = torch.zeros((nsample, nloc * nnei * 12), dtype=calc_dtype)
            rij_out = torch.zeros((nsample, nloc * nnei * 3), dtype=calc_dtype)
            nlist_out = torch.full((nsample, nloc * nnei), -1, dtype=torch.int32)

            # ---- coord 重排为 (nsample, nall, 3) 用于循环索引 ----
            coord_3d = coord_t.reshape(nsample, nall, 3)

            # ---- ntypes: davg/dstd 的第一维大小，用于 atom_type 越界保护 ----
            ntypes = int(davg_t.shape[0])

            for frame in range(nsample):
                coord_frame = coord_3d[frame]  # (nall, 3) torch tensor
                type_frame = type_t[frame]  # (nall,)   torch tensor

                for atom_idx in range(nloc):
                    atom_type_early = int(type_frame[atom_idx])
                    # type < 0 或 type >= ntypes 跳过（与 golden 保持一致）
                    if atom_type_early < 0 or atom_type_early >= ntypes:
                        continue

                    i_coord = coord_frame[atom_idx]  # (3,)

                    # 阶段1: 邻居查找 (torch 计算)
                    neighbors = []
                    for j_idx in range(nall):
                        if j_idx == atom_idx:
                            continue
                        if int(type_frame[j_idx]) < 0:
                            continue
                        diff = coord_frame[j_idx] - i_coord
                        nr2 = torch.sum(diff * diff)
                        nr_val = float(torch.sqrt(nr2))
                        if nr_val <= self.rcut_r:
                            neighbors.append((int(type_frame[j_idx]), nr_val, j_idx))

                    # 按 (type, distance) 升序排序
                    neighbors.sort(key=lambda x: (x[0], x[1]))

                    # 按 sel_a 分段填充 nlist
                    type_counts = [0] * len(self.sel_a)
                    for t_val, d_val, j in neighbors:
                        if (
                            t_val < len(self.sel_a)
                            and type_counts[t_val] < self.sel_a[t_val]
                        ):
                            out_idx = atom_idx * nnei + sec[t_val] + type_counts[t_val]
                            nlist_out[frame, out_idx] = j
                            type_counts[t_val] += 1

                    # 阶段2: 环境矩阵 (torch 计算)
                    atom_type = int(type_frame[atom_idx])
                    for ii in range(nnei):
                        idx_value = ii * 4
                        idx_deriv = ii * 12
                        nbor_idx = int(nlist_out[frame, atom_idx * nnei + ii])

                        if nbor_idx >= 0:
                            rr = coord_frame[nbor_idx] - i_coord
                            nr2 = torch.sum(rr * rr)
                            # nr2 极小值保护: 同 golden 函数，防止 inr4 溢出 float32 为 inf
                            # 导致 inr3=nan/inf 传播到 vv0-vv11。
                            # 标量 min 按 input dtype 取 float32/float64(1e-10)，
                            # 与 golden 的 tf.maximum(nr2, tf.constant(1e-10, dtype)) 一致。
                            safe_nr2 = torch.clamp(nr2, min=1e-10)
                            # rsqrt 必须用 torch.rsqrt: 标量 torch.rsqrt 与 golden 的
                            # tf.math.rsqrt 位级 100% 一致（5 万样本实测）；
                            # 禁止改写为 1.0 / torch.sqrt(...)（仅 86.6% 位一致，
                            # torch.sqrt/除法非全程 IEEE 正确舍入）。
                            inr = torch.rsqrt(safe_nr2)
                            nr = safe_nr2 * inr
                            inr2 = inr * inr
                            inr4 = inr2 * inr2
                            inr3 = inr4 * nr

                            nr_val = float(nr)
                            sw_f, dsw_f = _spline5_switch(
                                nr_val, self.rcut_r_smth, self.rcut_r
                            )
                            sw = torch.tensor(sw_f, dtype=calc_dtype)
                            dsw = torch.tensor(dsw_f, dtype=calc_dtype)

                            nr2_val = float(nr2)
                            rr0 = float(rr[0])
                            rr1 = float(rr[1])
                            rr2 = float(rr[2])
                            inr_f = float(inr)
                            inr2_f = float(inr2)
                            inr3_f = float(inr3)

                            # nr_val/nr2_val 除零保护: 原子坐标重合时 nr=0 导致 ZeroDivisionError
                            safe_nr_val = max(nr_val, 1e-10)
                            safe_nr2_val = max(nr2_val, 1e-10)
                            # 内层数学项先按 Python float64 计算, 再截断到计算 dtype
                            # (与 golden 的 tf.constant(expr, dtype) 语义一致),
                            # 之后与 sw/dsw 做计算 dtype 的乘减。
                            # 符号零对齐: TF 0-dim 标量算术 (mul/sub/div) 会把操作数中的
                            # -0.0 按 +0.0 处理 (实测 f32/f64 一致, 如 (-0.0)*正数 -> +0.0,
                            # 而 IEEE/torch 为 -0.0; golden 的 vv 项均为 0-dim 标量运算)。
                            # 为与 golden 位级一致, 两处可产生 -0.0 的操作数在进入乘法前
                            # 用 +0.0 归一化 (x + 0.0 仅把 -0.0 变 +0.0, 其余值位级不变):
                            #   1) t1 表达式 (如 2*rr_a*rr_b*inr2_f, rr 分量为精确 +0.0 时
                            #      与负分量相乘得 -0.0) —— Python 域归一;
                            #   2) dd*dsw (dd 为 +0.0 且 dsw<0 时得 -0.0) —— tensor 域归一。
                            # 减法本身无需处理: term1 归一后无 -0.0, term2 的 -0.0 仅来自
                            # (非负零)*+0.0, 两侧同值, x-(±0.0) 在 x 非 -0.0 时结果一致。
                            dd0 = torch.tensor(1.0 / safe_nr_val, dtype=calc_dtype) * sw
                            dd1 = (
                                torch.tensor(rr0 / safe_nr2_val, dtype=calc_dtype) * sw
                            )
                            dd2 = (
                                torch.tensor(rr1 / safe_nr2_val, dtype=calc_dtype) * sw
                            )
                            dd3 = (
                                torch.tensor(rr2 / safe_nr2_val, dtype=calc_dtype) * sw
                            )

                            vv0 = torch.tensor(
                                rr0 * inr3_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd0 * dsw + 0.0) * torch.tensor(
                                rr0 * inr_f, dtype=calc_dtype
                            )
                            vv1 = torch.tensor(
                                rr1 * inr3_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd0 * dsw + 0.0) * torch.tensor(
                                rr1 * inr_f, dtype=calc_dtype
                            )
                            vv2 = torch.tensor(
                                rr2 * inr3_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd0 * dsw + 0.0) * torch.tensor(
                                rr2 * inr_f, dtype=calc_dtype
                            )
                            vv3 = torch.tensor(
                                2.0 * rr0 * rr0 * inr2_f - inr_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd1 * dsw + 0.0) * torch.tensor(
                                rr0 * inr_f, dtype=calc_dtype
                            )
                            vv4 = torch.tensor(
                                2.0 * rr0 * rr1 * inr2_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd1 * dsw + 0.0) * torch.tensor(
                                rr1 * inr_f, dtype=calc_dtype
                            )
                            vv5 = torch.tensor(
                                2.0 * rr0 * rr2 * inr2_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd1 * dsw + 0.0) * torch.tensor(
                                rr2 * inr_f, dtype=calc_dtype
                            )
                            vv6 = torch.tensor(
                                2.0 * rr1 * rr0 * inr2_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd2 * dsw + 0.0) * torch.tensor(
                                rr0 * inr_f, dtype=calc_dtype
                            )
                            vv7 = torch.tensor(
                                2.0 * rr1 * rr1 * inr2_f - inr_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd2 * dsw + 0.0) * torch.tensor(
                                rr1 * inr_f, dtype=calc_dtype
                            )
                            vv8 = torch.tensor(
                                2.0 * rr1 * rr2 * inr2_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd2 * dsw + 0.0) * torch.tensor(
                                rr2 * inr_f, dtype=calc_dtype
                            )
                            vv9 = torch.tensor(
                                2.0 * rr2 * rr0 * inr2_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd3 * dsw + 0.0) * torch.tensor(
                                rr0 * inr_f, dtype=calc_dtype
                            )
                            vv10 = torch.tensor(
                                2.0 * rr2 * rr1 * inr2_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd3 * dsw + 0.0) * torch.tensor(
                                rr1 * inr_f, dtype=calc_dtype
                            )
                            vv11 = torch.tensor(
                                2.0 * rr2 * rr2 * inr2_f - inr_f + 0.0, dtype=calc_dtype
                            ) * sw - (dd3 * dsw + 0.0) * torch.tensor(
                                rr2 * inr_f, dtype=calc_dtype
                            )

                            vv_list = [
                                vv0,
                                vv1,
                                vv2,
                                vv3,
                                vv4,
                                vv5,
                                vv6,
                                vv7,
                                vv8,
                                vv9,
                                vv10,
                                vv11,
                            ]
                            dd_list = [dd0, dd1, dd2, dd3]

                            rij_out[frame, atom_idx * nnei * 3 + ii * 3 + 0] = rr0
                            rij_out[frame, atom_idx * nnei * 3 + ii * 3 + 1] = rr1
                            rij_out[frame, atom_idx * nnei * 3 + ii * 3 + 2] = rr2

                            # 归一化并存储 (Python float64 除法, 赋值时截断到计算 dtype,
                            # 与 golden 的 assign 语义一致)
                            base_davg = atom_type * ndescrpt + idx_value
                            for k in range(12):
                                descrpt_deriv[
                                    frame, atom_idx * nnei * 12 + idx_deriv + k
                                ] = float(vv_list[k]) / max(
                                    float(dstd_flat[base_davg + k // 3]), 1e-10
                                )
                            for k in range(4):
                                descrpt[frame, atom_idx * nnei * 4 + idx_value + k] = (
                                    float(dd_list[k]) - float(davg_flat[base_davg + k])
                                ) / max(float(dstd_flat[base_davg + k]), 1e-10)
                        else:
                            # 无邻居: descrpt = -davg/dstd (SE §5.2)
                            base_davg = atom_type * ndescrpt + idx_value
                            for k in range(4):
                                descrpt[frame, atom_idx * nnei * 4 + idx_value + k] = (
                                    -float(davg_flat[base_davg + k])
                                    / max(float(dstd_flat[base_davg + k]), 1e-10)
                                )

            return [descrpt, descrpt_deriv, rij_out, nlist_out]

    # GEIR remote dispatch needs an explicit provider dict.
    third_party = {"torch": ThirdPartyImpl}

    # float32 outputs use cross_check (rtol=1e-4, atol=1e-8);
    # int32 nlist uses binary_equal.
    tolerance = {
        "float32": {"standard": "cross_check"},
        "int32": {"standard": "binary_equal"},
    }
