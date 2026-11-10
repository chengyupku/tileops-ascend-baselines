"""CPU continuation with integrity gates; all device commands use remote.py."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
REL=HERE.relative_to(ROOT)
HELPER=ROOT/'TileOPs/src/tileops/kernels/ascend/elementwise/_logroots.py'
EXPECTED='995d5405dce1be237694944bcdff0988e922a86e86b0a309fe81585bda8b8b55'
def state(stage,**extra):
    (HERE/'post-search-status.json').write_text(json.dumps({'stage':stage,**extra},indent=2))
def run(script,*args):
    subprocess.run(['python3',str(HERE/script),*args],check=True)
try:
    state('waiting_for_initial_configuration_batch')
    while True:
        assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==EXPECTED,'Frozen helper changed'
        try:rows=json.loads((HERE/'complete-search-status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):rows=[]
        if len(rows)==20:
            assert all(row['rc']==0 for row in rows),'Initial configuration batch failed'
            break
        os.kill(1092674,0)
        time.sleep(5)
    state('repairing_first_four_cases_for_current_helper')
    run('complete_search_batch.py','--tag','config15-r2','--op','Log1pFwdOp',
        '--case','16M-float16','--case','16M-bfloat16','--case','16M-float32','--case','256M-float16')
    state('remeasuring_compiler_opponent_with_cached_compilation')
    plan=json.loads((HERE/'PLAN.json').read_text())
    args=['--phase','compile','--op','Log1pFwdOp']
    if 'torch-compile' not in plan['baseline_scope']:
        args+=['--case','16M-float16','--case','16M-bfloat16','--case','16M-float32']
    run('baseline_batch.py',*args)
    run('fetch.py','search','baseline','environment','edges')
    state('selecting_verified_per_case_configs')
    run('select_configs.py','--apply')
    files=['TileOPs/src/tileops/kernels/ascend/elementwise/'+name+'.py' for name in ('_logroots','log','log1p','sqrt','rsqrt')]
    files.append(str(REL/'frozen-v3.json'))
    subprocess.run(['python3',str(HERE.with_name('arithmetic9_20261027')/'sync.py'),*files],check=True)
    state('final_log1p_public_entry_measurement')
    run('remote.py','--label','final-Log1pFwdOp-v3','--timeout','600','--',str(REL/'pilot.py'),
        '--op','Log1pFwdOp','--output',str(REL/'final/final-Log1pFwdOp-v3.jsonl'))
    run('fetch.py','final','baseline','environment')
    run('audit.py')
    audit=json.loads((HERE/'audit.json').read_text())
    log1p=[row for row in audit['comparison'] if row['op']=='Log1pFwdOp']
    assert len(log1p)==5 and all(row['pass'] and row['pass_including_additional'] for row in log1p),log1p
    qualified=json.loads((HERE/'qualified-baselines.json').read_text())
    pairs={(row['case_id'],row['library']) for row in qualified if row['op']=='Log1pFwdOp'}
    expected={(row['case_id'],lib) for row in log1p for lib in ('ops-math','ops-nn','torch')}
    compile_cases=[row['case_id'] for row in log1p] if 'torch-compile' in plan['baseline_scope'] else ['16M-float16','16M-bfloat16','16M-float32']
    expected.update((case,'torch-compile') for case in compile_cases)
    assert expected<=pairs,('Missing qualified Log1p opponents',sorted(expected-pairs))
    # It resumes the original formal coordinator after recording its finished
    # baseline request. Neither coordinator has an active device child here.
    os.kill(1055813,signal.SIGCONT)
    state('original_formal_acceptance_resumed',comparison=log1p)
except BaseException as exc:
    state('stopped_for_review',error=repr(exc))
    raise
