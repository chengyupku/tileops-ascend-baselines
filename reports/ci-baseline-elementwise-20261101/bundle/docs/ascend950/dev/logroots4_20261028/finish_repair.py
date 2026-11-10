"""Revalidate frozen repaired sources, then resume the existing coordinator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REL = HERE.relative_to(ROOT)
parser = argparse.ArgumentParser()
parser.add_argument('--version', required=True)
args = parser.parse_args()


def state(stage, **extra):
    (HERE / 'post-search-status.json').write_text(json.dumps({'stage': stage, 'version': args.version, **extra}, indent=2))


def run(script, *argv):
    subprocess.run(['python3', str(HERE / script), *argv], check=True)


def remote(label, script, argv, timeout=600, allow_test_failure=False):
    result = subprocess.run(['python3', str(HERE / 'remote.py'), '--label', label,
                             '--timeout', str(timeout), '--', str(REL / script), *argv])
    if result.returncode and not (allow_test_failure and result.returncode == 1):
        raise RuntimeError((label, result.returncode))


try:
    frozen = json.loads((HERE / ('frozen-' + args.version + '.json')).read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
               for name, expected in frozen['sources'].items())
    state('remeasuring_repaired_public_entries')
    for op in ('Log1pFwdOp', 'LogFwdOp'):
        tag = op + '-' + args.version
        remote('final-' + tag, 'pilot.py', ['--op', op, '--output', str(REL / 'final' / ('final-' + tag + '.jsonl'))])
    state('revalidating_log1p_after_shared_helper_change')
    tag = 'Log1pFwdOp-' + args.version
    remote('edges-' + tag, 'edges.py', ['--op', 'Log1pFwdOp', '--output', str(REL / 'edges' / ('edges-' + tag + '.json'))])
    remote('pytest-' + tag, 'invoke_pytest.py', ['--kernel-module', 'log1p',
           'TileOPs/tests/ops/test_unary_math.py::test_log1p',
           'TileOPs/tests/ops/test_unary_math.py::test_log1p_edge', '-q',
           '--junitxml=' + str(REL / 'edges' / ('pytest-' + tag + '.xml'))], timeout=360, allow_test_failure=True)
    remote('profile-' + tag, 'diagnose.py', ['--op', 'Log1pFwdOp', '--output', str(REL / 'profiles' / tag)], timeout=240)
    run('fetch.py', 'final', 'baseline', 'environment', 'edges', 'profiles')
    run('audit.py')
    audit = json.loads((HERE / 'audit.json').read_text())
    rows = [r for r in audit['comparison'] if r['op'] in ('LogFwdOp', 'Log1pFwdOp')]
    assert len(rows) == 10 and all(r['pass'] and r['pass_including_additional'] for r in rows), rows
    qualified = json.loads((HERE / 'qualified-baselines.json').read_text())
    pairs = {(r['op'], r['case_id'], r['library']) for r in qualified}
    expected = {(r['op'], r['case_id'], lib) for r in rows
                for lib in (('ops-math', 'ops-nn', 'torch') if r['op'] == 'Log1pFwdOp' else ('ops-math', 'torch'))}
    plan = json.loads((HERE / 'PLAN.json').read_text())
    compile_cases = [r['case_id'] for r in rows if r['op'] == 'Log1pFwdOp'] if 'torch-compile' in plan['baseline_scope'] else ['16M-float16', '16M-bfloat16', '16M-float32']
    expected.update(('Log1pFwdOp', case, 'torch-compile') for case in compile_cases)
    assert expected <= pairs, sorted(expected - pairs)
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash
               for name, expected_hash in frozen['sources'].items())
    # Existing finish_baselines.py is already waiting; do not create a second one.
    os.kill(1026045, signal.SIGCONT)
    state('original_formal_acceptance_resumed', comparison=rows)
except BaseException as exc:
    state('stopped_for_review', error=repr(exc))
    raise
