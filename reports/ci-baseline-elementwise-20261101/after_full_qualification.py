"""Finish prepared checker correction only after immutable full-suite export."""
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
state_path = BASE/'ci-full-v3/measurements/STATE.json'
while True:
    state = json.loads(state_path.read_text())
    if state['completed'] == state['expected'] == 70 and state['status'] != 'running':
        assert state['status'] == 'passed', state
        if (BASE/'export-after-full-01/PRIVATE_REPORT.json').exists():
            break
    time.sleep(20)

out = BASE/'FINAL_QUALIFICATION_STATE.json'
result = dict(status='running', completed=[])
out.write_text(json.dumps(result,indent=2)+'\n')
try:
    for script in ['apply_eager_helper_fix.py', 'run_eager_requalification.py',
                   'readmit_eager_helper.py', 'run_public_final.py']:
        print(json.dumps(dict(phase='final_qualification', script=script, status='starting')), flush=True)
        subprocess.run([sys.executable,str(BASE/script)], cwd=BASE,check=True)
        result['completed'].append(script)
        out.write_text(json.dumps(result,indent=2)+'\n')
    result.update(status='ready_to_commit_and_export', verified_operators=70, verified_cases=527)
except BaseException as exc:
    result.update(status='failed', error=str(exc))
    raise
finally:
    out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
