"""CPU-only one-shot reload after an existing guarded pair ends normally."""
import fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools'))
from run_baseline_cache import identity,put
host=sys.argv[1];assert host in ('local','server2')
R=ROOT/'reports/baseline-cache-20261027';D=R/'inspection-20261028T121544';H=R/host
config=json.loads((D/(host+'-reload-config.json')).read_text());expected=config['supervisor']
state=dict(started_at=time.time(),identity=identity(os.getpid()),previous_supervisor=expected)
def record(phase,**extra):
    state.update(at=time.time(),phase=phase,**extra);put(D/(host+'-reload-state.json'),state)
def authorized():
    return ((H/'STOP').exists() and (H/'STOP').read_text()==config['token'] and not (R/'STOP').exists()
            and hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest()==config['plan_sha256']
            and hashlib.sha256((ROOT/'tools/run_baseline_cache.py').read_bytes()).hexdigest()==config['runner_sha256'])
while True:
    if not authorized(): record('held_control_or_source_changed');raise SystemExit(0)
    try: running=identity(expected['pid'])==expected
    except FileNotFoundError: running=False
    if not running:break
    record('waiting_for_normal_pair_boundary');time.sleep(2)
live=json.loads((H/'LIVE.json').read_text())
if live.get('identity')!=expected or live.get('phase')!='stopped_by_control':
    record('held_unexpected_exit',live_identity=live.get('identity'),live_phase=live.get('phase'));raise SystemExit(0)
with (H/'supervisor.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not authorized():record('held_control_changed_at_boundary');raise SystemExit(0)
    subprocess.run([sys.executable,str(D/'apply_exact_holds.py'),host],check=True)
    (H/'STOP').unlink()
if host=='local':
    subprocess.run(['systemctl','start','baseline-cache-local-20261027.service'],check=True)
else:
    with (D/'server2-reloaded-supervisor.log').open('a') as log:
        p=subprocess.Popen(['bash',str(ROOT/'tools/baseline_cache_service.sh'),'--campaign',str(R),'--host',host],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['cpu_wrapper']=identity(p.pid)
for _ in range(30):
    live=json.loads((H/'LIVE.json').read_text());current=live.get('identity')
    try: valid=current!=expected and identity(current['pid'])==current
    except (FileNotFoundError,KeyError,TypeError):valid=False
    if valid:record('reloaded',supervisor=current,live_phase=live.get('phase'));raise SystemExit(0)
    time.sleep(1)
record('reload_identity_not_observed');raise SystemExit(1)
