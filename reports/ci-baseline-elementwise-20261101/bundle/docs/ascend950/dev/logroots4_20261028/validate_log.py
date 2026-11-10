"""Complete correctness and profiling after the bounded performance search ended."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REL = HERE.relative_to(ROOT)
p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
a = p.parse_args()
tag = 'LogFwdOp-' + a.version
frozen = json.loads((HERE / ('frozen-' + a.version + '.json')).read_text())


def state(stage, **extra):
    (HERE / ('log-' + a.version + '-validation-status.json')).write_text(json.dumps(dict(stage=stage, **extra), indent=2))


def run(script, *args):
    subprocess.run(['python3', str(HERE / script), *args], check=True)


try:
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in frozen['sources'].items())
    state('checking_retained_final_results')
    run('fetch.py', 'final', 'baseline', 'environment')
    run('audit.py')
    rows = [r for r in json.loads((HERE / 'audit.json').read_text())['comparison'] if r['op'] == 'LogFwdOp']
    assert len(rows) == 5 and all(r['tileops_us'] is not None for r in rows), rows
    state('checking_integrated_numerics')
    run('remote.py', '--label', 'edges-' + tag, '--timeout', '600', '--', str(REL / 'edges.py'),
        '--op', 'LogFwdOp', '--output', str(REL / 'edges' / ('edges-' + tag + '.json')))
    result = subprocess.run(['python3', str(HERE / 'remote.py'), '--label', 'pytest-' + tag,
                             '--timeout', '360', '--', str(REL / 'invoke_pytest.py'), '--kernel-module', 'log',
                             'TileOPs/tests/ops/test_unary_math.py::test_log',
                             'TileOPs/tests/ops/test_unary_math.py::test_log_edge', '-q',
                             '--junitxml=' + str(REL / 'edges' / ('pytest-' + tag + '.xml'))])
    assert result.returncode in (0, 1), result.returncode
    run('remote.py', '--label', 'profile-' + tag, '--timeout', '240', '--', str(REL / 'diagnose.py'),
        '--op', 'LogFwdOp', '--regex', '16M-(bfloat16|float32)', '--output', str(REL / 'profiles' / tag))
    run('fetch.py', 'final', 'edges', 'profiles')
    state('complete', pytest_return_code=result.returncode, performance='one_case_deferred_at_0.7971428214; original gate failure retained')
except BaseException as exc:
    state('stopped_for_review', error=repr(exc))
    raise
