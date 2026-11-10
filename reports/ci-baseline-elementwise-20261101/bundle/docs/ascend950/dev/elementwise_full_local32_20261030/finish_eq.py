"""Serial final-source Eq verification; each device job gets its own guard."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

r = Path(__file__).resolve().parent
root = r.parents[3]
snapshot = json.loads((r / 'eq-final-v2-sources.json').read_text())['source_manifest']

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
        f.write('\n' + datetime.datetime.now(datetime.timezone.utc).isoformat() + ' MAIN Eq ' + message + '\n')
    print(message, flush=True)

for label in ['eq-shared-public-v1', 'eq-broadcast-v2']:
    receipt(label)
jobs = [
    ('eq-public-v2', 'public_eq.py', None, 19),
    ('eq-manifest-v2', 'measure_eq.py', None, 14),
    ('eq-numerical-v2', 'eq_numerical.py', 'numerical', 27),
    ('eq-bool-probe-v2', 'eq_bool_probe.py', 'bool-probe', 65536),
    ('eq-bool-broadcast-v2', 'eq_bool_broadcast.py', 'bool-broadcast', 6),
    ('eq-special-v2', 'eq_special_values.py', 'special-values', 9),
    ('eq-channel-v2', 'eq_channel_stress.py', 'channel-stress', 18),
    ('eq-ub-v2', 'eq_ub_regression.py', 'ub-regression', 12),
    ('eq-boundaries-v1', 'eq_boundaries.py', 'boundaries', 90),
    ('eq-tail-v1', 'eq_tail_stress.py', 'tail-stress', 30),
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
        args += ['--output', str(r / 'eq-manifest-v2.jsonl')]
    subprocess.run(args, cwd=root, check=True)
    frozen()
    receipt(label)
    if suffix:
        result = json.loads((r / f'eq-{suffix}.json').read_text())
        if suffix == 'bool-probe':
            assert result['pairs'] == count and result['op_mismatches'] == 0 and result['input_unchanged']
        else:
            assert len(result['checks']) == count and all(x['passed'] for x in result['checks'])
        assert result['runtime_limit'] == {'cube_core_num': 32, 'vector_core_num': 64}
        assert result['module_sha256'] == hashlib.sha256((root / 'TileOPs/src/tileops/kernels/ascend/elementwise/comparison.py').read_bytes()).hexdigest()
        for path, expected in result['dependency_manifest'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    elif script.startswith('measure_'):
        rows = [json.loads(line) for line in (r / 'eq-manifest-v2.jsonl').read_text().splitlines()]
        assert len(rows) == count
        assert all(x['correct'] and x['input_unchanged'] and x['eager']['latency_ms'] / x['latency_ms'] >= .8 for x in rows)
    else:
        import xml.etree.ElementTree as ET
        suites = list(ET.parse(r / ('maximum-public.xml' if script == 'public_maximum.py' else 'eq-public.xml')).getroot().iter('testsuite'))
        assert sum(int(s.get('tests', 0)) for s in suites) == count
        assert all(int(s.get('errors', 0)) + int(s.get('failures', 0)) + int(s.get('skipped', 0)) == 0 for s in suites)
    record(label + ' 完成，guard有效/冻结源码一致。')

frozen()
subprocess.run([sys.executable, str(r / 'audit_eq.py')], cwd=root, check=True)
p = r / 'PLAN.json'
plan = json.loads(p.read_text())
assert 'EqFwdOp' not in plan['local32_completed']
plan['local32_completed'].append('EqFwdOp')
plan.update(current_op='NeFwdOp', phase='ne_preparation', active_request=None,
            session_id=None, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            last_verified='Eq audited:14 manifest,19 public,12 UB,18 broadcast,27 numerical,9 special,90 boundaries,30 tail,18 channel,65536 bool pairs,6 bool broadcast')
p.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
record('最终审计通过，PLAN/ledger44/70本地验证，28登记baselinepair待补。报告eq-REPORT.json。下一Ne准备，尚未启动。全目标active。')
