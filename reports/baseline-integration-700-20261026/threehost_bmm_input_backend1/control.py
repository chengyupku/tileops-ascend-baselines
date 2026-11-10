from pathlib import Path
import subprocess,json,shlex,time,datetime
P=Path(__file__).resolve().parent
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/'+P.name
OLD=R.replace(P.name,'threehost_bmm1')
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
HOST='openlibing-DevEnv-661495'
def hb(phase):
 (P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,accepted_added=0)))
def remote(cmd,host=HOST):
 r=subprocess.run([SSH,host,cmd],capture_output=True,text=True,timeout=60)
 return dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
def py(src):return remote('python3 -c '+shlex.quote(src))
def upload(name):
 r=subprocess.run([SSH,HOST,'cat > '+shlex.quote(R+'/'+name)],input=(P/name).read_bytes(),capture_output=True);assert r.returncode==0,r.stderr
if __name__=='__main__':
 hb('three_host_preflight')
 from concurrent.futures import ThreadPoolExecutor
 commands=[('local',None),('server1',HOST),('server2','openlibing-DevEnv-055409')]
 def status(item):
  name,host=item
  if host: r=remote('npu-guard status --json',host)
  else:
   t=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True);r=dict(rc=t.returncode,stdout=t.stdout,stderr=t.stderr)
  (P/(name+'_status.json')).write_text(json.dumps(r,indent=2));return name,r
 with ThreadPoolExecutor(3) as pool:
  for name,r in pool.map(status,commands): print(name,r)
 r=py(f'''from pathlib import Path
import json,shutil,platform
p=Path({OLD!r})
print(json.dumps(dict(machine=platform.machine(),disk=shutil.disk_usage(p)._asdict(),paths={{x:(p/x).exists() for x in ['TileOPs/scripts/ascend_pair_check.py','TileOPs/workloads/gemm.py','tools','private_provider/op_api/lib/libcust_opapi.so','resource_gate.py','run.sh']}},cpu_entry=Path('/opt/tileops-ci/runtime/bin/python').exists(),new_path_exists=Path({R!r}).exists())))
print(Path('/opt/tileops-all-pairs/control/private_python.py').read_text())
print((p/'resource_gate.py').read_text())
''')
 (P/'REMOTE_PREFLIGHT_RAW.json').write_text(json.dumps(r,indent=2));print(r)
