"""Serial final-source Ge verification; each device job gets its own guard."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

r = Path(__file__).resolve().parent
root = r.parents[3]
snapshot = json.loads((r / 'ge-final-sources.json').read_text())['source_manifest']

def frozen():
    for path, expected in snapshot.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path

def receipt(label):
    guard = json.loads((r / 'requests' / label / 'guard.json').read_text())
    assert guard['exit_code'] == 0 and guard['performance_valid'], label
    assert 'race remains' not in (r / 'requests' / label / 'output.log').read_text(), label

def record(message):
    with (root / '.agent_memory/PROJECTS/tileops-ascend950/STATE.md').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write('\n' + datetime.datetime.now(datetime.timezone.utc).isoformat() + ' MAIN Ge ' + message + '\n')
    print(message, flush=True)

for label in ['ge-public-v1', 'ge-manifest-v1']:
    receipt(label)
rows=[json.loads(l) for l in (r/'ge-manifest-v1.jsonl').read_text().splitlines()]
assert len(rows)==13 and all(v['correct'] and v['eager']['latency_ms']/v['latency_ms']>=.8 for v in rows)
jobs = [
    ('ge-broadcast-v2', 'ge_broadcast_regression.py', 'broadcast-regression', 18),
    ('ge-numerical-v1', 'ge_numerical.py', 'numerical', 27),
    ('ge-bool-probe-v1', 'ge_bool_probe.py', 'bool-probe', 65536),
    ('ge-bool-broadcast-v1', 'ge_bool_broadcast.py', 'bool-broadcast', 6),
    ('ge-special-v1', 'ge_special_values.py', 'special-values', 9),
    ('ge-channel-v1', 'ge_channel_stress.py', 'channel-stress', 18),
    ('ge-ub-v1', 'ge_ub_regression.py', 'ub-regression', 12),
    ('ge-boundaries-v1', 'ge_boundaries.py', 'boundaries', 90),
    ('ge-tail-v1', 'ge_tail_stress.py', 'tail-stress', 30),
]
for label, script, suffix, count in jobs:
    frozen()
    p = r / 'PLAN.json'
    plan = json.loads(p.read_text())
    plan.update(phase=label.replace('-', '_'), active_request=label,
                updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    record('启动 ' + label + '，串行父session见PLAN.session_id。')
    args = [sys.executable, str(r / 'run_local.py'), label, script]
    subprocess.run(args, cwd=root, check=True)
    frozen()
    receipt(label)
    if suffix:
        result = json.loads((r / f'ge-{suffix}.json').read_text())
        if suffix == 'bool-probe':
            assert result['pairs'] == count and result['op_mismatches'] == 0 and result['input_unchanged']
        else:
            assert len(result['checks']) == count and all(x['passed'] for x in result['checks'])
        assert result['runtime_limit'] == {'cube_core_num': 32, 'vector_core_num': 64}
        assert result['module_sha256'] == hashlib.sha256((root / 'TileOPs/src/tileops/kernels/ascend/elementwise/comparison.py').read_bytes()).hexdigest()
        for path, expected in result['dependency_manifest'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    record(label + ' 完成，guard有效/冻结源码一致。')

frozen()
subprocess.run([sys.executable, str(r / 'audit_ge.py')], cwd=root, check=True)
p = r / 'PLAN.json'
plan = json.loads(p.read_text())
assert 'GeFwdOp' not in plan['local32_completed']
plan['local32_completed'].append('GeFwdOp')
plan.update(current_op='LeFwdOp', phase='le_preparation', active_request=None,
            session_id=None, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            last_verified='Ge audited:13 manifest,13 public,12 UB,18 broadcast,27 numerical,9 special,90 boundaries,30 tail,18 channel,65536 bool pairs,6 bool broadcast')
p.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
record('最终审计通过，PLAN/ledger48/70本地验证，26登记baselinepair待补。报告ge-REPORT.json。下一Le准备，尚未启动。全目标active。')
