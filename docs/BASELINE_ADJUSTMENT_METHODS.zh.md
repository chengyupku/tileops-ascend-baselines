# Baseline 性能调整方法

截至 2026-11-10。目标是让 baseline 忠实体现现有库的可达性能：调整顶层适配和调用方式，使用已有完整 API 或编译原表达式；不修改原 kernel、编译器、参考答案、检查器或容差。当前复测已暂停。

## 已采用的方法

| 方法 | 具体做法与实例 | 适用边界 |
| --- | --- | --- |
| 保留广播，避免物化重复数据 | Block quant/dequant 将数据 reshape 成块，用 `scale[..., None]` 广播，避免把 scale 展开到每个元素；SGL Mul 按共享 scale 的轴重排，复用原 `fused_scale_shift` API。 | 保持实际广播语义、块边界、输出顺序和 dtype；必要的转置、连续化、补零和裁剪都计时。 |
| 把逐段调用改为批量调用 | 均匀 MeanPooling reshape 后一次 mean；变长池化按实际长度分组，或使用完整 `segment_reduce` API；GroupedGemm 使用 `npu_grouped_matmul`。 | 实际 offsets、indices、group sizes 仍参与运行。保留尾段真实分母、空组和请求顺序；完整调用更少不保证更快。 |
| 使用语义匹配的完整原生 API | FP16/BF16 INT8 dequant 使用 `npu_anti_quant`；部分无 running statistics 的 InstanceNorm 使用每通道一组的 GroupNorm。 | 逐 case 核对舍入、精度和副作用。FP32 dequant 不套用上述 antiquant 路线；带 running statistics 的 InstanceNorm 不套用上述替换。 |
| 用 `torch.compile` 融合现有计算 | 分别评估原 Torch 表达式编译，以及原生 API 包装层编译；减少部分逐元素中间结果和独立提交。 | 两类实现分别标记。使用 `fullgraph=True`、不静默吞掉编译错误；原量化码和浮点检查照常执行。完整图不等于一个 kernel。 |
| 删除无实际作用的布局操作 | 仅在 padding 大小为零时跳过 `pad`；保留确实需要的 padding。已用于 block quant/dequant，也试用于 SGL Add/Sub。 | 不省略必要的 dtype 转换。TileLang 路线仍有未通过的编译或适配尝试，不能算作已交付收益。 |

这些方法可以组合，但必须测量完整包装层。例如 block dequant 的原生路线仍包含分块补齐、reshape、原生反量化和输出裁剪；不能只计原生内核时间。

## 已验证的代表结果

下表是同一专项内的 **baseline 对 baseline** 设备时间对照，每个变体均复核了 15 个窗口；不是 TileOPs 加速比，也不表示整类算子均有相同收益。

| 精确实例 | 原路径 → 调整后（μs） | 主要变化 |
| --- | ---: | --- |
| SGL Mul，`linear-bias-float16` | 341.39 → 12.45 | 共享广播 scale 的布局 |
| MeanPooling，`uniform-batched-float16-float32` | 267.03 → 23.17 | 逐段 mean 改为批量归约 |
| GroupedGemm，`qwen3-30b-ep8-empty-bfloat16` | 344.59 → 25.96 | 完整 grouped matmul API |
| AdaLayerNormZero，`llama-8b-decode-bfloat16` | 15.40 → 6.18 | Torch 表达式编译 |
| InstanceNorm，`image-affine-bfloat16` | 15.28 → 8.41 | 不带 running statistics 的等价完整 API |

Quant/dequant 已完成 71 个 case、160 个正式变体的验证。其中 FP8 per-token 的编译路线相对原 eager 约快 3.91–4.92 倍，FP8 block 的布局调整约快 1.63–2.13 倍。其它高加速比专项暂停时为 46 个正式变体，尚未全量完成。

## 不应强行采用的调整

- **编译不一定正确或更快。** 部分 INT8、FP8、INT4 编译结果不满足原始量化码检查，已拒绝；INT4 per-group 和 INT8 per-channel 的最终选择仍保留原路径。
- **原生 API 不一定更快。** GroupedGemm 的 wgrad case 从原路径约 800.72 μs 变为原生列表 API 约 1077.47 μs，因此保留更快实现。
- **编译不自动消除回退。** Cumprod 的已测编译路径仍有 AICPU；当前 InstanceNorm 编译尝试还出现主机 CPU 回退。主机 CPU 与 AICPU 要分别记录，缺少设备任务记录不能填成零耗时。
- **接口存在不等于二进制可用。** 绑定实际加载的源码、动态库 SHA、架构和工具链；不能从源码注册推断旧 `.so` 已包含该 API。环境修复在私有进程或目录中进行，不改共享栈。

## 编译并行与验收

编译并行改善的是实验周转时间，不是 baseline 稳态延时。本轮使用 4 个 CPU compiler worker；线程数按实际 CPU 配额和内存预算设置，不按名义核心数或可见逻辑 CPU 数直接启动同等数量的 case。仅已确认不访问设备的编译阶段可在锁外并行；图捕获、首次执行及可能初始化 NPU 的导入仍须受主机 guard 管理，设备测量串行。

资源预算还须检查显式中间张量。输入体积乘 16 只是估算，不能当实测峰值；本轮 SSDChunkState 有一个参考中间张量单独就需 128 GiB，超过 96 GiB 限额，该 case 已暂缓。

执行顺序为：核对合同和实际调用 → CPU 检查适配 → guarded pilot → 正式旧新对照 → 原始窗口重放 → 按精确 case 择优。正式口径保留物理 32 AIC/64 AIV、3 次 warmup、15 个窗口及每窗口 1 GiB flush；flush 不等于已证明目标缓存完全冷却。所有输入相关 metadata、布局及输出转换计入调用；有状态输入按公开 reset 合同恢复。失败、未测和试测不得混为正式通过。

## 证据入口

- 已发布的 [quant/dequant 报告](../reports/quant-baseline-fix-server2-20261110/REPORT.json)、[精确 case 注册表](../reports/quant-baseline-fix-server2-20261110/verified_cases.json)及 [适配源码](../reports/quant-baseline-fix-server2-20261110/quant_baselines/)。
- 其它专项的本地工作区记录：`reports/high-speedup-baseline-fix-server2-20261110/RESULTS.json`、`SOURCE_IDENTITY_AUDIT.json`、`PAUSE_RECEIPT.json`。上述新增结果尚未作为完整专项发布，原始测量归档不上传 GitHub。
- 通用复现约束见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
