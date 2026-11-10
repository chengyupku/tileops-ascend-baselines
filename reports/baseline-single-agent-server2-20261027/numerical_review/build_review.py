"""Read existing evidence only; never import a device runtime or edit verdicts."""
from pathlib import Path
import csv
import datetime
import hashlib
import json

ROOT = Path('/root/dyq/workspace')
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'reports/baseline-integration-700-20261026'
rows = json.loads((OUT.parent / 'PAIR_STATUS.json').read_text())
rows = [r for r in rows if r['status'] == 'original_numerical_failure']
assert len(rows) == 18

groups = {
    'fla_fp16_decode': {
        'count': 7,
        'assessment': '倾向原生 decode 精度边界；未发现明确适配错误，未定位到具体内核指令。',
        'comparison': '本批成功账本中，同算子同 npu_chunk_gated_delta_rule_fwd API 有 22 份 verified prefill 回执；这些成功不能证明 decode 同样达到更严格的阈值。',
        'audit': '核对实测适配器：QKV 转 BNSD，gate/beta 保持原 API 要求的 [B,T,Hv]，FP32 无损扩展，initial clone，scale=K**-0.5，输出为 BSND；这 7 个 case 无 L2/beta/raw-gate 变换。',
        'pattern': '各 batch 都只有少数输出越过 6e-7 阈值，最大误差约 9.54e-7 至 3.81e-6。原 checker 已包含相邻舍入值容许，不能靠改变容差消除失败。',
        'limitations': '大部分 decode 使用 NPU FP32 reference；prefill 成功使用 CPU FP64 reference。未保存这 7 次完整真实输入输出，不能证明每个误差都来自哪一步舍入。',
        'sources': [
            str(OLD / 'fla/integration700_fla_native.py'),
            str(OLD / 'threehost_linear49/integration700_fla_native.py'),
            str(ROOT / 'ascend_baselines/flash-linear-attention-npu/torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py'),
            str(ROOT / 'TileOPs/workloads/linear_attention/gdn.py'),
        ],
    },
    'fla_recurrent_bf16': {
        'count': 1,
        'assessment': '倾向原生 L2/beta recurrence 数值精度差异；具体根因未证实。',
        'comparison': '该 case 实际调用 npu_recurrent_kda，不能拿 chunk prefill 成功当作同路径成功。',
        'audit': '核对 BF16 输入、标量 gate 广播、初始状态、BSND、scale，以及 use_qk_l2norm/use_beta_sigmoid/allow_neg_eigval 三个 flag 的原 API 直传；K=V=128，无实质 feature padding。',
        'pattern': '首个输出比较已通过，失败位置为 FP32 final_state：13/131072 个元素，最大差 7.45e-8，阈值 4e-8。',
        'limitations': '仅确定最终状态不满足固定 checker，不能据此声称整个 BF16 算子失效。',
        'sources': [str(ROOT / 'TileOPs/benchmarks/ascend_all_pairs/fla/providers.py'), str(ROOT / 'TileOPs/workloads/linear_attention/gdn.py')],
    },
    'ops_transformer_chunk_kda': {
        'count': 7,
        'assessment': '较强证据指向原生 ChunkKdaFwd 的 FP16/短序列路径，而非一般的布局、gate、beta 或状态传递错误；具体内核根因未定位。',
        'comparison': '同 baseline GDN 有 79 份原始 checker 状态为 verified 的成功回执，另 1 份账本 accepted 的 checker 为 source_changed，未当作严格成功。DeltaNet 有 11 份 verified。成功调用 ChunkGatedDeltaRule 或 RecurrentKda，均不同于这 7 个失败的 ChunkKdaFwd 路径。',
        'audit': '逐项核对 ACLNN 头文件顺序、ctypes 类型、BSND、gate 自然对数广播、beta FP32 扩展、initialState FP32、cuSeqlens、stateVFirst、scale 和可选输出。DeltaNet 以精确零 log-gate 表示无门控。',
        'pattern': '4 个 GDN decode 和 1 个 DeltaNet decode 输出近乎全错；2 个 ragged 部分输出失败。保存的 grouped-ragged 真实输入输出显示：仅 token 0（长度为 1 的序列）750 个输出失败，后续 token 和最终状态通过。',
        'capture_evidence': 'gdn_fp16_capture14 的 bound_inputs.pt 与 inputs.pt 检查显示 Q/K/V、gate、beta、initial 完全一致。额外同路径 decode-b1 对照也失败。此次 CPU 分析排除了简单的漏 scale/漏初始状态解释，不是新的 NPU 验证，也不取代 checker。',
        'limitations': '实参捕获覆盖 grouped-ragged 与同路径 decode-b1 对照；其余 case 是实测适配器与共同错误模式对照，未宣称逐个定位内核根因。',
        'sources': [
            str(OLD / 'surge_dense_linear92/worker05/package/binding.py'),
            str(OLD / 'surge_dense_linear92/worker05/package/direct_acl.py'),
            str(OLD / 'threehost_linear49/package_local/binding_fp16_delta.py'),
            str(ROOT / 'ascend_baselines/ops-transformer/attention/chunk_kda_fwd/docs/aclnnChunkKdaFwd.md'),
            str(OLD / 'gdn_fp16_capture14/attempt02_varlen/CPU_ANALYSIS.json'),
            str(OLD / 'gdn_fp16_capture14/attempt02_varlen/LOCALIZATION.json'),
            str(OUT / 'CHUNK_KDA_CAPTURE_CPU.json'),
        ],
    },
    'ops_nn_instance_norm': {
        'count': 2,
        'assessment': '暂未发现适配错误，倾向原生数值实现与固定 reference 的差异；具体误差来源仍待定。',
        'comparison': '同算子有 9 条保留的历史成功，不能全部当作新审计成功。追查其中 3 份真实 checker 均 verified，但 API 为 GroupNormV2，原生编译/派发证据有历史缺口。其他算子有 3 份 LayerNormV4/AdaLayerNorm、1 份 BatchNormV3/BatchNorm verified 回执。',
        'audit': 'eval：推理模式、NCHW 通道、running mean/variance、epsilon/momentum、affine 和完整输出。wider：对每个通道的全空间做 LayerNorm，权重/偏置在空间复制，stack 后恢复 NCHW，无跨通道归约。',
        'pattern': 'eval 在原始 FP16 接口与无损 FP32 扩展接口都出现相同 855 个失败；wider 在 GroupNormV2、BatchNormV3、LayerNormV4 三种历史调用中都仅同一个元素失败、误差 0.00390625。不能只怪最后一次 LayerNorm 适配器。',
        'limitations': '没有这些历史失败的真实输入输出捕获，无法进一步区分原生内部精度、CPU reference 实现与最终 FP16 舍入。保留失败，不修 baseline、不修改 reference/容差。',
        'sources': [
            str(OLD / 'nn_remote_remaining6/case3/adapter.py'),
            str(OLD / 'surge_dense_linear92/worker03/instance_eval_fp32_readmission/adapter.py'),
            str(ROOT / 'TileOPs/workloads/norm.py'),
            str(ROOT / 'reports/baseline-integration-20261026/nn_norm/InstanceNormFwdOp--wider-channel-affine-float16/output/checker.json'),
            str(ROOT / 'reports/baseline-integration-20261026/nn_norm/InstanceNormFwdOp--wider-channel-affine-float16--batchnorm/output/checker.json'),
            str(ROOT / 'reports/baseline-integration-20261026/nn_norm/InstanceNormFwdOp--image-eval-float16/output/checker.json'),
        ],
    },
    'tilelang_round': {
        'count': 1,
        'assessment': '已定位原生 PTO round 路径的半整数舍入语义不匹配；适配器未改写 round 算法。',
        'comparison': 'FP16 只有历史保留成功记录，不是同一次 PTO/FP32 扩展路径已验证成功的证据；FP32 记录为执行异常，不能说所有 dtype 数值失败。',
        'audit': '实际保留的生成代码在 FP32 tile 上调用 TCVT(..., RoundMode::CAST_ROUND)；原 compiler 将 T.tile.round 降为 CAST_ROUND，PTO A5 分发 RoundAType。canonical reference 是 torch.round(decimals=0)，半整数取偶数。',
        'pattern': '例如 0.5：CAST_ROUND 为 1、torch.round 为 0；2.5 分别为 3 与 2。与真实失败的 2764/1048576、最大误差恰好 1 相符。',
        'limitations': '保存的代码证实语义不一致；未保存真实失败张量，因此未逐元素证明 2764 个差异全部是 tie。CPU 例子仅解释语义，不冒充设备重跑。修改 compiler 或选择不同数学舍入模式会改变被测 baseline 行为，本轮不做。',
        'sources': [
            str(OLD / 'tla_simple_tilelang/element_adapter_bounded.py'),
            str(OLD / 'tla_simple_tilelang/runs/RoundFwdOp_16M-bfloat16_v6/jit/launch-evidence/tilelang-2889389-gk893yfd/ca1f12715b91125cfdd6a541b95ad359f6ed6928e2322a528bc3646959b32e24.kernel.txt'),
            str(ROOT / 'ascend_baselines/tilelang-ascend/src/target/codegen_ascend_pto.cc'),
            str(ROOT / 'ascend_baselines/tilelang-ascend/3rdparty/pto-isa/include/pto/npu/a5/TCvt.hpp'),
            str(ROOT / 'TileOPs/workloads/elementwise.py'),
            str(OUT / 'ROUND_TIE_SEMANTICS.json'),
        ],
    },
}

reviewed = []
for r in rows:
    lib, op, cid = r['key']
    group = ('fla_recurrent_bf16' if 'bfloat16' in cid else 'fla_fp16_decode') if lib == 'flash-linear-attention-npu' else {
        'ops-transformer': 'ops_transformer_chunk_kda', 'ops-nn': 'ops_nn_instance_norm', 'tilelang-ascend': 'tilelang_round'
    }[lib]
    p = Path(r['checker'])
    d = json.loads(p.read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == r['checker_sha256'], p
    assert d['status'] == 'failed' and d['stage'] == 'verify'
    reviewed.append({'key': r['key'], 'group': group, 'checker': str(p), 'checker_sha256': r['checker_sha256'],
                     'api': d.get('api'), 'error': d['error'], 'assessment': groups[group]['assessment'],
                     'confirmed_adapter_bug': False, 'original_failure_retained': True, 'new_device_run': False})

hashes = {}
for g in groups.values():
    for s in g['sources']:
        p = Path(s)
        hashes[s] = hashlib.sha256(p.read_bytes()).hexdigest()

report = {
    'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': '二号远端原分配 81 pair 中的 18 个数值失败；仅对此范围作结论',
    'method': '同 baseline / 算子 / 原 API / dtype / case 模式比较；核对真实 checker、实测适配代码和可用捕获；tilelang-ascend 最后审查。',
    'summary': {'reviewed': 18, 'confirmed_adapter_bugs': 0, 'new_npu_runs': 0, 'new_passes': 0,
                'located_semantic_mismatch': 1, 'likely_native_path_or_precision_limit': 15,
                'nn_numeric_cause_unresolved': 2, 'verdicts_changed': 0},
    'interpretation': '证据倾向主要是 baseline 特定路径的能力/精度边界，不能等同于已证明 18 个都是 baseline 内核 bug；各算子并非所有 case 都失败。',
    'constraints': ['unsupported case 不再对接', '保持原始失败和成功证据', '不改变 baseline 算法/内核/compiler/精度/调度', '不改变输入、seed、reference、checker 或容差', '不进行没有修正条件的设备重跑', '仅一个 agent，无派遣'],
    'groups': groups, 'pairs': reviewed, 'reviewed_source_hashes': hashes,
}
(OUT / 'NUMERICAL_REVIEW.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
with (OUT / 'PAIR_REVIEW.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(['baseline', 'operator', 'case_id', 'group', 'assessment', 'checker'])
    for r in reviewed:
        w.writerow([*r['key'], r['group'], r['assessment'], r['checker']])
state = json.loads((OUT / 'STATE.json').read_text())
state.update(phase='review_complete', completed_at=report['updated_at'], summary=report['summary'], report=str(OUT / 'NUMERICAL_REVIEW.json'))
(OUT / 'STATE.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report['summary'], ensure_ascii=False))
