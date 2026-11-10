"""Remeasure the repaired public entry and resume only after all acceptance gates."""
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
p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
a = p.parse_args()


def run(script, *args):
    subprocess.run(['python3', str(HERE / script), *args], check=True)


def state(stage, **extra):
    (HERE / 'post-search-status.json').write_text(json.dumps({'stage': stage, **extra}, indent=2))


try:
    frozen = json.loads((HERE / ('frozen-' + a.version + '.json')).read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in frozen['sources'].items())
    state('repaired_log1p_public_measurement', version=a.version)
    label = 'final-Log1pFwdOp-' + a.version
    run('remote.py', '--label', label, '--timeout', '600', '--', str(REL / 'pilot.py'),
        '--op', 'Log1pFwdOp', '--output', str(REL / 'final' / (label + '.jsonl')))
    run('fetch.py', 'final', 'baseline', 'environment')
    run('audit.py')
    audit = json.loads((HERE / 'audit.json').read_text())
    rows = [r for r in audit['comparison'] if r['op'] == 'Log1pFwdOp']
    assert len(rows) == 5 and all(r['pass'] and r['pass_including_additional'] for r in rows), rows
    qualified = json.loads((HERE / 'qualified-baselines.json').read_text())
    pairs = {(r['case_id'], r['library']) for r in qualified if r['op'] == 'Log1pFwdOp'}
    expected = {(r['case_id'], lib) for r in rows for lib in ('ops-math', 'ops-nn', 'torch')}
    plan = json.loads((HERE / 'PLAN.json').read_text())
    compile_cases = [r['case_id'] for r in rows] if 'torch-compile' in plan['baseline_scope'] else ['16M-float16', '16M-bfloat16', '16M-float32']
    expected.update((case, 'torch-compile') for case in compile_cases)
    assert expected <= pairs, sorted(expected - pairs)
    os.kill(1055813, signal.SIGCONT)
    state('original_formal_acceptance_resumed', version=a.version, comparison=rows)
    run('finish_baselines.py')
except BaseException as exc:
    state('stopped_for_review', error=repr(exc))
    raise
