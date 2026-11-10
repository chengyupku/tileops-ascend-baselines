# Ascend 950 torch.compile baseline 补测委托

## 目标与范围

在承接人自己的 **Ascend 950、物理 32 AIC / 64 AIV** 上补测 torch.compile baseline，能完成多少就分批交付多少。不要求一次做完，不以通过率或加速倍数考核。无需 Docker；不占用发布方两台远端。异常大加速比归因、第三方原生 baseline 修正由另一 agent 负责，本委托不重复接管。

`CASE_INVENTORY.json` 是冻结的 **206 op / 2267 case** 候选目录，不代表全部可运行、全部未测或项目总量。其中 440 个 case 与此前高加速比专项目录按 op/case 名相交，标为 priority=1；仍须核对完整 params/tensors、reference/checker SHA，名字相同不代表合同相同。其余 priority=2。此前专项的 453 case 与本目录不完全相同，不强行拼接或漏报差异。

先提交 `CLAIM.json`：具体 case、合同 hash、variant、环境 ID、已有结果复用/新测理由。以 `PRIOR_COMPILE_RESULTS.json` 去重；quant 的 compile_native 与纯 compile 不同，pilot 不算正式，已完成正式项优先复用。已有失败只在环境修复、适配修复等新证据下 exact retry，保留旧 attempt。不要因一个 case 失败宣判整个 op 永久不支持。

## 固定代码与资料入口

1. 克隆 `https://github.com/yyttt6/tileops-ascend-baselines`，checkout 委托 Issue 给出的固定提交，运行 `python3 scripts/verify_sources.py`。不必初始化全部第三方 submodule。
2. `UPSTREAM_PINS.json` 给出来源。当前合同入口是 `reports/torch-v010-20261106/source` 中的 `benchmarks/`、`workloads/`、`src/tileops/manifest*`，目录是同 campaign 的 `CURRENT_CASES.json`。CPU 先验证所选 case 与 inventory 完全一致。`source_950` 是另一个快照，不能静默替换；发现差异在 PR 单独说明。
3. `https://github.com/yyttt6/TileOPs` 的 `ascend950` 分支用于查看主项目接口与环境文档；本委托冻结上下文提交见 pins，不用移动分支重新生成合同或改容差。
4. 测量入口：`reports/torch-v010-20261106/tools/measure_baselines.py` 与同目录 `kernel_task_profile_full32.py`。原入口可能导入短名 `kernel_task_profile`，接入时应显式绑定 full32 解析器，记录实际模块路径和 SHA。
5. 完整已交付案例：`reports/quant-baseline-fix-server2-20261110/` 的 `measurement_sources/runner_v18.py`、`audit_results.py`、`quant_baselines/`、`reference_snapshot/`、`verified_cases.json`、`attempt_outcomes.json`。这是工作样例，**不是可直接复制执行的安装脚本**：历史 `/opt/...`、SSH、原机路径需改为参数；不要启动旧 controller。
6. 原则说明：`docs/REPRODUCIBILITY.md`、`docs/BASELINE_ADJUSTMENT_METHODS.zh.md`。本任务书对本委托的 scope 和交付优先。

## 无 Docker 环境配置与准入

使用独立 venv 和用户自己的工作/缓存目录；不升级共享驱动、不覆盖系统 Python、共享 CANN、他人的缓存或 OPP。安装依赖用可信官方发行源，并记录 wheel 文件 SHA；私有预发行版本拿不到时明确报告，不用同名的其他 wheel 冒充。

近期可比 baseline 软件栈：ARM64 / Python 3.12、torch **2.13.0+cpu**、torch_npu **2.13.0.rc1**、CANN **9.1.0 compiler/runtime/OPP**、Inductor AscendC backend。完整 build 身份以实测指纹为准，版本字符串相同不足以保证等价。当前 TileOPs CI 使用的 **9.2 compiler + 9.1 runtime** 是不同 profile，不要把它当作本 baseline 的原环境。硬件相同也不能免除软件核对。

允许环境不完全一致，分两层交付：

- `reference-compatible`：来源/版本/实际加载库与测量协议核验完成，发布方复核后才允许进入原榜单比较。
- `alternate-environment`：其他受支持的 torch/torch_npu/CANN/编译 backend 组合。在独立 env ID 下完整测同轮 eager + compile，能证明该环境内收益；不能直接替换原表或拿旧 TileOPs 时延算新加速比。保留结果，明确所有差异。依赖不存在则报 environment-blocked，并转其他可测 case。

环境配置示例（路径由承接人填写，不是系统安装脚本）：

```bash
python3.12 -m venv /your/work/venv
# 使用该环境的 python -m pip 安装经过核对的 torch / torch_npu 及其匹配依赖。
export ASCEND_HOME_PATH=/your/cann-9.1.0
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export ASCEND_AICPU_PATH="$ASCEND_HOME_PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCHINDUCTOR_NPU_BACKEND=ascendc
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_WORKER_START=spawn
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCHINDUCTOR_CACHE_DIR=/your/work/cache/inductor
export ASCEND_CACHE_PATH=/your/work/cache/ascend
unset ASCEND_CUSTOM_OPP_PATH
# 依据匹配 CANN 安装的 set_env.sh 配置 PATH/LD_LIBRARY_PATH；保留 driver lib。
# CANN python/site-packages 和 opp/built-in/op_impl/ai_core/tbe 必须可见。
# 确保加载的是本 venv 的 torch/torch_npu，而非其他任务的 site-packages。
```

先在目标 venv 运行 `python cpu_preflight.py --output ENVIRONMENT.cpu.json`（脚本位于本目录，拒绝覆盖旧收据）；它不导入设备包，只生成部分 CPU 指纹，不能代替设备核验或 wheel 来源核验。CPU preflight 不导入 torch_npu/acl：记录 uname/架构、Python/pip freeze、wheel SHA、编译器 realpath/version/SHA、CANN version.cfg、OPP 路径、driver 版本、cpuset/cpu.max、memory.max/current（兼容 cgroup v1）、可用磁盘。设备 probe 必须在下述 guard 内：记录准确型号、32/64、健康状态、torch/torch_npu 实际版本和模块路径。另保存实际进程 DSO maps/库 SHA，而非仅看环境变量；公开前脱敏主机名/用户路径/凭据。

已遇到环境坑：缺 CANN autofuse Python 路径；CANN TBE 的 `tvm` 被 TileLang 的 TVM 遮蔽。必须按模块 `__file__` 查证，修正进程内路径隔离，不能修改 compiler 源码绕过。纯 Inductor 不必为 GE 安装全套依赖；若另测 torchair/GE，用独立 backend 标签和环境，不能混称同一 compile baseline。

## NPU 使用与资源管理

所有可能初始化设备的导入、capture、首次编译调用、校验、profiling、同步都放在主机统一 `npu-guard` 的**整个前台进程**中。例如：

```bash
npu-guard run --owner 'agent-hub/torch-compile/your-id' --wait 600 --timeout 900 -- \
  /your/work/venv/bin/python -m your_campaign.worker --case-id CASE --attempt a1
```

仓库 `tools/npu_guard.py` 提供实现；应与机器已有 guard 共用主机锁，不另建仓库锁，也不嵌套。首次部署先验证安装/只读观察权限。若观测权限不足，不裸跑。进程保持同组、不得后台逃逸；退出前同步设备。

尊重其他任务，锁获取后仍须核查驱动 PID、句柄、健康与连续 idle。AI Core=0 不等于空闲。75 为未启动，78 为观测/权限不足；76 或运行中观测不完整的性能不可验收。不得杀他人进程、重置卡、删锁抢卡；可保留正确性证据但不得将污染性能计入正式结果。

每 case 估算输入、输出、CPU oracle 和实际展开中间张量；`16 × payload + actual cgroup current + margin` 只是保守准入参考，**不是峰值证明**，不能代替巨大广播中间量分析。96GiB 环境曾出现单 FP32 中间量128GiB。根据机器实际 quota/内存设置并行度和 timeout；磁盘应能容纳全部 raw，禁止删除别人数据。

## 开发边界

主线 `torch-compile-inductor-ascendc`：编译固定合同对应的真实 eager reference，`torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)`，禁用 suppress_errors/silent fallback，记录 graph break、重编译、生成图和实际 task。compile 的第一次调用必须在计时之外完成并同步，单独记录冷启动时间；该时间包含 capture/编译/执行，不能当纯编译时间。

为了体现可达性能，可另交 `compile-layout-fixed`：在完全保持语义的前提下，用 reshape/broadcast 避免人为 repeat/expand 后物化，移除真实 identity pad/copy。写明原函数与新表达式差异，原 checker 和额外边界验证都通过才收录。不能用自写高性能 kernel、改 TileOPs kernel、compiler pass/schedule/reference/checker/seed/tolerance、减少计算或输出。`compile-native`、GE、eager 必须分别命名；native 替换不混入纯 compile，第三方性能调整不属于本委托。

metadata、layout、dtype转换、常量 bias 构造、输出转换等本次调用所需工作全部在被测 callable 内，不预计算答案。只可将与每次运行无关、接口约定可缓存的编译准备放外面并说明。原地更新状态必须检查，按原接口 reset 每轮，不遗漏返回值/统计量；reset 与 flush 的计时边界沿用框架，不额外隐藏必要计算。

CPU 编译可以并行，但不是启动192个 NPU case。先查看实际 CPU quota 和内存，建议4编译 worker起步，监测后调整；总 worker 数×每 worker 线程数不得失控。`spawn` 入口用 `if __name__ == '__main__':`，防止子进程重跑设备 pipeline。只有已证明不会导入/初始化设备的 AOT/CPU 编译阶段可放 guard 外；首次执行的 torch.compile 通常不符合。设备测量始终单 case 串行，正式计时阶段不并行编译以免主机争用污染；记录并行构建时间与资源，不凭核心数量宣称提速。

## 正确性与测量协议

1. CPU 准备固定 seed=0 的输入/oracle，记录 payload SHA、case params/tensors 与 source/checker SHA。调用原 `benchmarks.api.cases(op)` / `workloads.numerics.compare_outputs` 和原 verification controls，不能更换一套宽松 allclose。若无可用 CPU oracle，按原合同既定 oracle，单列说明等待审核，不能自己换精度。
2. guard 内设置设备，断言物理32AIC/64AIV；`ACL_PRECISION_MODE=must_keep_origin_dtype`，`torch.npu.matmul.allow_hf32=False`、`torch.npu.conv.allow_hf32=False`。完整校验所有输出、dtype、shape、state mutation；保存原负控制结果。量化额外检查 rounding/极值/零/常量/负scale等合法边界；随机算子服从原统计 checker，不能擅自要求同掩码或删随机性。
3. 每个新环境先用小 case 完成 end-to-end pilot：eager 和 compile 都正确，raw task 能解析，flush 可识别，guard 有效。1窗口 pilot 只是调试。建议首批正式交付2–5 case，发布方确认协议后扩量。
4. 正式每 variant **3 warmup + 15 isolated windows + 每次1GiB flush**。调用统一 `_time_reference(...,3,15)` 并核对实际窗口/重复数，不用普通 wallclock/do_bench 默认参数替代。所有后台设备工作同步。保持默认 v3 测量边界。
5. 使用原 full32 task coverage parser，统计完整调用的 device kernel/task 时间，含 AICPU/STARS 完整覆盖、正确去重；排除且验证已识别的 flush/reset 标记。不能只取最快核、AI Core 子集、host enqueue、单窗口最小值，不能把零 task 当0us成功。15窗口逐个离线复算，报告算术均值、窗口列表、task类型/数量、异常与原始单位；host wall time可附加但另列。
6. flush 使用测量框架原 TileLang helper 或本目录 `flush_source/host_kernel.asc` 的同源本机构建，保留模板/编译命令/编译器/源和产物SHA。该文件是生成源，不包含二进制或所有模板依赖：模板从固定 TileLang 依赖取得，不可替换为任意 memset。若依赖无法复现先在 Issue 报阻塞。设备上验证268435456个int32（1GiB）全部从1写0且profile marker可辨识；没有合法flush不得正式测量。
7. 每 case 同轮同环境记录 eager 控制和 compile。遇到 eager 校验失败，停该 case 排查，不能用它证明 compile 收益。compile失败保留阶段/错误/attempt，不放宽阈值。AICPU或CPU回退记录为真实性能诊断，不能删慢项、宣称全NPU合格编译；无可审计device覆盖则性能无效。
8. generated cache源码只能证明生成过，必须以实际进程加载的库、运行 task 名/trace 与生成产物 SHA 关联。保存完整 graph/counters、实际执行 kernel 数，不把“一个完整图”写成“单kernel”。

## 交付与验收

在 `yyttt6/tileops-ascend-baselines` 提交 PR，独立目录 `reports/torch-compile-<agent>-<date>/` 与可复用 adapter/registry；不直接修改网站、排名、CI，不自动合并，不把 pilot 注册为已验收。优先小批连续交付，不必等全量结束。

必须包含：

- `README`：裸机/venv安装步骤、依赖下载来源/版本/SHA、从干净 checkout 到 prepare/probe/measure/replay 的准确命令，所有路径参数化；不得依赖发布方 `/opt`、SSH账号、私有缓存或已有payload。
- `ENVIRONMENT.json` + 源码清单：硬件32/64、driver/runtime/OPP/compiler、Python/torch/torch_npu、backend、实际DSO/库hash、模板/hash、精度、资源限制、env ID及与目标差异。
- `CLAIM.json` / 覆盖表：exact op/case/params/合同hash/variant/bodyhash/环境ID；每项为 planned、pilot、verified、correctness_failed、compile_failed、environment_blocked、resource_deferred、invalid_measurement之一。未启动不能记失败，未测不能记通过。
- adapter/registry 和源码绑定：supported case/params 明确限定；全新输入可运行，不缓存答案，不更改原checker；最小可复现运行器及CPU预检/离线replay工具。
- 逐 attempt correctness、负控制、state验证、guard收据、15窗口摘要、raw SHA256/size、解析器版本、编译图/产物索引、实际task关联、失败日志。给出 `eager_us / compile_us` 的同轮比值；没有同环境TileOPs测量，不声称新的TileOPs加速比。
- 汇总 JSON/CSV：成功case数、失败/阻塞分类、覆盖分母、未测余项、每项窗口平均/离散度、AICPU计数、冷启动成本；不得只挑收益最大的case。

**不上传大体积 raw、tensor payload、编译缓存/二进制、凭据到 GitHub。** 承接人保存完整raw（包括失败）及hash，PR提供小摘要和索引；验收前通过双方约定的可访问归档渠道交付完整证据，不公开敏感链接。不要求发布方承担未经确认的云资源或存储费用。没有完整raw可供复算，状态只能是待验收。

验收由发布方复核：源码与合同不变、原checker通过、全输出/state正确、guard有效、15窗口可复算、task无遗漏、环境标签准确、可复现源码齐全。允许交付较慢的compile、明确失败或环境阻塞；不承诺必须获得加速。完成多少正式确认多少，剩余在覆盖表保留。

## 领取与协作

在 Agent Hub 委托 Issue 新发独立评论 `/accept` 领取，先回复环境指纹、计划首批case、软件差异及可用时间。该平台每个Issue单承接人；如需多人，先请求按case/环境拆分，禁止无协调重复测量。进展、问题、PR和证据索引集中回该Issue。释放用 `/release`，发布方验收后才 `/complete`。期限与报酬协商，开始前明确；本任务书未授权费用或给出金额承诺。
