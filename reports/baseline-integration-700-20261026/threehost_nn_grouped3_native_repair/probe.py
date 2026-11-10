from pathlib import Path
import subprocess,concurrent.futures,json,datetime
P=Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
W='/opt/tileops-all-pairs/workspace'; OLD=W+'/reports/baseline-integration-700-20261026/threehost_nn11'
cmd='''uname -m
LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver python3 /opt/tileops-all-pairs/workspace/tools/npu_guard.py status --json
python3 - <<'INNER'
from pathlib import Path
import json
w=Path('/opt/tileops-all-pairs/workspace');p=w/'reports/baseline-integration-700-20261026/threehost_nn11';b=w/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build'
print('BUILD_EXISTS',b.exists())
if b.exists():
 print('BUILD_TOP',[x.name for x in b.iterdir()]);print('CACHE', '\\n'.join(l for l in (b/'build/CMakeCache.txt').read_text().splitlines() if any(s in l.lower() for s in ['soc','compute','tiling','binary','kernel','op_names','build_type','ascend','package'])))
for f in (p/'vendor/allpairs_nn').rglob('*mat_mul*v3*.json'):
 print('METADATA',f, f.stat().st_size);d=json.loads(f.read_text());print(str(d)[:6000])
for n in ['UPSTREAM_ORIGINAL_JIT_REUSE.json','UPSTREAM_NATIVE_READY.json']:
 f=p/n
 if f.exists(): print(n,f.read_text()[:12000])
INNER'''
def run(host):
 r=subprocess.run([SSH,host,cmd],capture_output=True,text=True); (P/(host+'.txt')).write_text(r.stdout+'\nSTDERR\n'+r.stderr);return host,r.returncode,len(r.stdout),r.stdout[:2600]
with concurrent.futures.ThreadPoolExecutor() as ex:
 for r in ex.map(run,['openlibing-DevEnv-661495','openlibing-DevEnv-055409']):print(r)
r=subprocess.run(['python3','tools/npu_guard.py','status','--json'],capture_output=True,text=True);(P/'local_STATUS.json').write_text(r.stdout)
(P/'HEARTBEAT.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'CPU package audit','accepted_added':0}))
