"""Wait for the guarded candidate search, then compare actual public calls."""
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(HERE.parents[3])


def state(stage, **values):
    (HERE / 'public-lut-v5-status.json').write_text(json.dumps(dict(stage=stage, **values), indent=2))


def run(script, *args):
    subprocess.run(['python3', str(HERE / script), *args], check=True)


try:
    state('waiting_for_full_table_search')
    while True:
        paths = list((HERE / 'requests').glob('optimize-log-bf16-lut-v5_*/status.json'))
        if paths:
            receipt = max((json.loads(p.read_text()) for p in paths), key=lambda r: r['completed_at'])
            break
        time.sleep(5)
    run('fetch.py', 'search', 'edges')
    assert receipt['guard_exit_code'] == 0, receipt
    version = 'v5'
    result = json.loads((HERE / 'search/log-bf16-lut-v5/result.json').read_text())
    if result['search_latency_ms'] >= .0155:
        state('candidate_rejected', reason='No clear improvement beyond current borderline result', result=result['search_latency_ms'])
        raise SystemExit(0)
    state('public_configuration_comparison', candidate=version)
    for case in ('16M-bfloat16', '256M-bfloat16'):
        run('remote.py', '--label', 'public-log-bf16-lut-' + version + '-' + case,
            '--timeout', '600', '--', str(REL / ('search_public_log_bf16_lut_' + version + '.py')),
            '--op', 'LogFwdOp', '--case', case)
    run('fetch.py', 'search', 'edges')
    state('ready_for_review', candidate=version)
except Exception as exc:
    state('stopped_for_review', error=repr(exc))
    raise
