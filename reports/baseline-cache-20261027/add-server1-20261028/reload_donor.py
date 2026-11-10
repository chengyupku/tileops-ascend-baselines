"""Keep current guarded pair intact, then reload the published exact ownership."""
import json,hashlib,os,sys,time,subprocess,fcntl
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools'))
from run_baseline_cache import identity,put
host=sys.argv[1];assert host in ('local','server2')
R=ROOT/'reports/baseline-cache-20261027';H=R/host;D=R/'add-server1-20261028';expected=json.loads((D/(host+'-boundary.json')).read_text())['identity']
plan_hash=hashlib.sha256((D/'PLAN.proposed.json').read_bytes()).hexdigest();token='add-server1-20261028/ownership-boundary\n'
state={'started_at':time.time(),'previous_supervisor':expected,'identity':identity(os.getpid())}
def record(phase,**kw):state.update(phase=phase,at=time.time(),**kw);put(D/(host+'-reload.json'),state)
def authorized():return (H/'STOP').exists() and (H/'STOP').read_text()==token and not (R/'STOP').exists() and hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest()==plan_hash
while True:
    if not authorized():record('held_control_or_plan_changed');raise SystemExit(0)
    try:alive=identity(expected['pid'])==expected
    except FileNotFoundError:alive=False
    if not alive:break
    record('waiting_normal_boundary');time.sleep(2)
live=json.loads((H/'LIVE.json').read_text());assert live['identity']==expected and live['phase']=='stopped_by_control'
with (H/'supervisor.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert authorized();(H/'STOP').unlink()
if host=='local':subprocess.run(['systemctl','start','baseline-cache-local-20261027.service'],check=True)
else:
    with (D/'server2-supervisor.log').open('a') as f:
        p=subprocess.Popen(['bash',str(ROOT/'tools/baseline_cache_service.sh'),'--campaign',str(R),'--host',host],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);state['wrapper']=identity(p.pid)
for _ in range(30):
    live=json.loads((H/'LIVE.json').read_text());current=live.get('identity')
    try:ok=current!=expected and identity(current['pid'])==current
    except (FileNotFoundError,KeyError,TypeError):ok=False
    if ok:record('reloaded',supervisor=current);raise SystemExit(0)
    time.sleep(1)
record('new_identity_not_observed');raise SystemExit(1)
