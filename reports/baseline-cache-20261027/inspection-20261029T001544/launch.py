import fcntl, hashlib, json, os, pathlib, subprocess, sys, time
h=sys.argv[1];R=pathlib.Path('reports/baseline-cache-20261027').resolve();H=R/h;D=R/'inspection-20261029T001544'
assert json.loads((R/'PLAN.json').read_text())['assignment_revision']==7
assert (R/'PLAN.json').read_bytes()==(D/'PLAN.revision7.json').read_bytes()
with (H/'supervisor.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (H/'STOP').read_text()=='inspection-20261029T001544/rebalance-pending-only\n'
    (H/'STOP').unlink()
cmd=['bash','tools/baseline_cache_service.sh','--campaign',str(R),'--host',h]
log=(D/(h+'-supervisor.log')).open('ab')
p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'at':time.time(),'host':h,'wrapper_pid':p.pid,'command':cmd,'plan_sha256':hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest(),'unlimited':True}
(D/(h+'-LAUNCH.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
