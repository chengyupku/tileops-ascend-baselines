"""Serial final-source Minimum verification; each device job gets its own guard."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

r = Path(__file__).resolve().parent
root = r.parents[3]
snapshot = json.loads((r / 'minimum-final-v2-sources.json').read_text())['source_manifest']

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
        f.write('\n' + datetime.datetime.now(datetime.timezone.utc).isoformat() + ' MAIN Minimum ' + message + '\n')
    print(message, flush=True)

for label in ['minimum-numerical-v2', 'minimum-bool-broadcast-v1']:
    receipt(label)

jobs = [
    ('minimum-bool-probe-v2', 'minimum_bool_probe.py', 'bool-probe', 65536),
    ('minimum-public-v1', 'public_minimum.py', None, 24),
    ('minimum-shared-maximum-v1', 'public_maximum.py', None, 26),
    ('minimum-manifest-v2', 'measure_minimum.py', None, 14),
    ('minimum-special-v1', 'minimum_special_values.py', 'special-values', 9),
    ('minimum-channel-v1', 'minimum_channel_stress.py', 'channel-stress', 18),
    ('minimum-ub-v1', 'minimum_ub_regression.py', 'ub-regression', 12),
    ('minimum-broadcast-v1', 'minimum_broadcast_regression.py', 'broadcast-regression', 18),
    ('minimum-boundaries-v1', 'minimum_boundaries.py', 'boundaries', 90),
    ('minimum-tail-v1', 'minimum_tail_stress.py', 'tail-stress', 30),
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
    if script.startswith('measure_'):
        args += ['--output', str(r / 'minimum-manifest-v2.jsonl')]
    subprocess.run(args, cwd=root, check=True)
    frozen()
    receipt(label)
    if suffix:
        result = json.loads((r / f'minimum-{suffix}.json').read_text())
        if suffix == 'bool-probe':
            assert result['pairs'] == count and result['op_mismatches'] == 0 and result['input_unchanged']
        else:
            assert len(result['checks']) == count and all(x['passed'] for x in result['checks'])
        assert result['runtime_limit'] == {'cube_core_num': 32, 'vector_core_num': 64}
        assert result['module_sha256'] == hashlib.sha256((root / 'TileOPs/src/tileops/kernels/ascend/elementwise/arithmetic_extrema.py').read_bytes()).hexdigest()
        for path, expected in result['dependency_manifest'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    elif script.startswith('measure_'):
        rows = [json.loads(line) for line in (r / 'minimum-manifest-v2.jsonl').read_text().splitlines()]
        assert len(rows) == count
        assert all(x['correct'] and x['input_unchanged'] and x['eager']['latency_ms'] / x['latency_ms'] >= .8 for x in rows)
    else:
        import xml.etree.ElementTree as ET
        suites = list(ET.parse(r / ('maximum-public.xml' if script == 'public_maximum.py' else 'minimum-public.xml')).getroot().iter('testsuite'))
        assert sum(int(s.get('tests', 0)) for s in suites) == count
        assert all(int(s.get('errors', 0)) + int(s.get('failures', 0)) + int(s.get('skipped', 0)) == 0 for s in suites)
    record(label + ' 完成，guard有效/冻结源码一致。')

frozen()
subprocess.run([sys.executable, str(r / 'audit_minimum.py')], cwd=root, check=True)
p = r / 'PLAN.json'
plan = json.loads(p.read_text())
assert 'MinimumFwdOp' not in plan['local32_completed']
plan['local32_completed'].append('MinimumFwdOp')
plan.update(current_op='EqFwdOp', phase='eq_preparation', active_request=None,
            session_id=None, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            last_verified='Minimum audited:14 manifest,24 public,12 UB,18 broadcast,27 numerical,9 special,90 boundaries,30 tail,18 channel,26 shared Maximum,65536 bool pairs,6 bool broadcast')
p.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
record('最终审计通过，PLAN/ledger43/70本地验证，56登记baselinepair待补。报告minimum-REPORT.json。下一Eq准备，尚未启动。全目标active。')
