"""One foreground process runs exact cases sequentially, with one guard per case."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / 'TLA28_PLAN.json').read_text())
results = []
for item in plan:
    row = dict(item)
    if item['action'] != 'run_once_minimal_original_api':
        row.update(status=item['action'], verified=False, newly_executed=False)
        results.append(row)
        (HERE / 'TLA28_RESULTS.json').write_text(json.dumps(results, indent=2)+'\n')
        continue
    attempt = item['attempt_id']
    out = HERE / attempt
    if (out / 'guard.json').exists():
        raise RuntimeError('Refusing an unchanged rerun: ' + attempt)
    out.mkdir(exist_ok=True)
    print(json.dumps({'event':'start', 'attempt':attempt, 'key':item['key'], 'timeout':120}), flush=True)
    (HERE / 'TLA28_CURRENT.json').write_text(json.dumps(row, indent=2)+'\n')
    cmd = [sys.executable, str(HERE/'run_guarded.py'), attempt, '120',
           sys.executable, str(HERE/'task_exec.py'), 'bash', str(HERE/'local-python'),
           str(HERE/'run_tla28_case.py'), attempt, *item['key'][1:]]
    with (out/'launcher.log').open('w') as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    receipt = json.loads((out/'guard.json').read_text())
    meta = receipt.get('guard_metadata') or {}
    checker_path = out/'checker.json'
    checker = json.loads(checker_path.read_text()) if checker_path.exists() else None
    if checker:
        status = 'checker_' + checker.get('status', 'unknown')
    elif rc == 124:
        status = ('device_sync_timeout_skip' if (out/'HOST_RETURNED.json').exists()
                  else 'guarded_command_timeout_skip')
    elif rc in (75, 78):
        status = 'guard_unavailable_not_verified'
    else:
        status = 'command_failed_without_checker'
    row.update(status=status, newly_executed=bool(meta.get('child')), guard_exit=rc,
               verified=False, guard=str(out/'guard.json'),
               checker=str(checker_path) if checker else None,
               host_returned=(out/'HOST_RETURNED.json').exists(),
               device_synchronized=(out/'DEVICE_SYNCHRONIZED.json').exists(),
               conflict_detected=meta.get('conflict_detected'),
               visibility_lost=meta.get('visibility_lost'),
               checker_stage=checker.get('stage') if checker else None,
               checker_error=(checker.get('error','')[:1200] if checker else None),
               finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'RESULT.json').write_text(json.dumps(row, indent=2)+'\n')
    results.append(row)
    (HERE/'TLA28_RESULTS.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'event':'finished', 'attempt':attempt, 'status':status,
                      'guard_exit':rc, 'handled':len(results), 'total':28}), flush=True)
(HERE/'TLA28_CURRENT.json').write_text('{"phase":"finished","active_command":null}\n')
print('TLA28_SEQUENCE_FINISHED', flush=True)
