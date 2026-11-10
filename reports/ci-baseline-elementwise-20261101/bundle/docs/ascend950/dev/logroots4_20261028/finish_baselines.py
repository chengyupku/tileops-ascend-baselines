"""CPU continuation after public acceptance, retaining all measurement gates."""
import json
import os
from pathlib import Path
import subprocess
import time
HERE=Path(__file__).resolve().parent
def state(stage,**extra):
    (HERE/'remaining-baselines-status.json').write_text(json.dumps({'stage':stage,**extra},indent=2))
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def run(script,*args):
    subprocess.run(['python3',str(HERE/script),*args],check=True)
try:
    state('waiting_for_public_acceptance')
    while True:
        rows=read(HERE/'formal-batch-v1-status.json') or []
        if len(rows)==28:
            assert all(row['rc']==0 or row['label'].startswith('pytest-') for row in rows),rows
            break
        continuation=read(HERE/'post-search-status.json') or {}
        assert continuation.get('stage')!='stopped_for_review',continuation
        os.kill(1026045,0)
        time.sleep(5)
    state('checking_primary_results')
    run('fetch.py','final','baseline','environment','edges','profiles')
    run('audit.py')
    audit=read(HERE/'audit.json')
    assert audit['own_valid_cases']==20 and audit['cases_passing_including_additional']==20,audit['comparison']
    for phase in ('tla','catlass'):
        state('measuring_'+phase)
        run('baseline_batch.py','--phase',phase)
    plan=read(HERE/'PLAN.json')
    if 'torch-compile' in plan['baseline_scope']:
        state('measuring_remaining_compiler_opponents')
        run('baseline_batch.py','--phase','compile','--op','LogFwdOp','--op','SqrtFwdOp','--op','RsqrtFwdOp')
    state('auditing_complete_scope')
    run('fetch.py','final','baseline','environment','edges','profiles')
    run('audit.py')
    run('timing_report.py')
    audit=read(HERE/'audit.json')
    state('ready_for_final_review',own_valid_cases=audit['own_valid_cases'],
          formal_pairs=audit['formal_valid_baseline_pairs'],additional_pairs=audit['additional_valid_baseline_pairs'],
          passing=audit['cases_passing'],passing_including_additional=audit['cases_passing_including_additional'],
          unattempted=audit['baseline_pairs_unattempted'],unqualified=audit['baseline_pairs_without_valid_measurement'])
except BaseException as exc:
    state('stopped_for_review',error=repr(exc))
    raise
