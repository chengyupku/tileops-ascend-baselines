import json,pathlib,hashlib,subprocess,concurrent.futures,datetime
W=pathlib.Path('/root/dyq/workspace'); O=W/'reports/baseline-integration-700-20261026/threehost_pto_historical23'; S=json.loads((O/'SCOPE.json').read_text())
script='''import pathlib,json,hashlib,socket
W=pathlib.Path(ROOT)
def norm(p):
 p=str(p)
 for prefix in ['/root/dyq/workspace/','/opt/tileops-all-pairs/workspace/']:
  if p.startswith(prefix):p=p[len(prefix):]
 return W/p
def read(p):
 p=norm(p)
 if not p.is_file():return {'path':str(p),'exists':False}
 b=p.read_bytes();r={'path':str(p),'exists':True,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 if p.suffix in ['.json','.py','.txt'] and len(b)<120000:r['text']=b.decode(errors='replace')
 return r
rows=[]
for s in SCOPE:
 p=norm(s['checker_receipt']);r={'key':[s['baseline'],s['operator'],s['case_id']],'checker':read(p),'guard':read(p.parent/'guard.json'),'native':read(p.parent/'native.json'),'guard_running':read(p.parent/'GUARD_RUNNING.json'),'artifacts':[]}
 if p.exists():
  c=json.loads(p.read_text())
  for f,h in c.get('source_sha256',{}).items():
   a=read(f);a['historical_sha256']=h;a['matches_historical']=a.get('sha256')==h;r['artifacts'].append(a)
 rows.append(r)
base='reports/baseline-integration-700-20261026/tla_simple_tilelang/'
extra=[base+x for x in ['run_case.py','run_v7.py','gemm_adapter_pto.py','gemm_rest_pto.guard.json','gemm_probe.guard.json','REGISTERED.json','integration700_tla_simple_tilelang_gemm7.py','integration700_tla_simple_tilelang_gemm_rest7.py','frozen_gemm7/private_tla/native.py','frozen_gemm_rest7/private_tla/native.py']]
extra+=['TileOPs/benchmarks/ascend_all_pairs/tla/'+x for x in ['__init__.py','native.py','integration700_tla_simple_tilelang_gemm7.py','integration700_tla_simple_tilelang_gemm_rest7.py']]
extra+=['.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package/tilelang/jit/adapter/libgen.py']
print(json.dumps({'hostname':socket.gethostname(),'root':str(W),'rows':rows,'extra':[read(f) for f in extra]}))
'''
def go(host):
 root=str(W) if host=='local' else '/opt/tileops-all-pairs/workspace'
 code='ROOT='+repr(root)+'\nSCOPE='+repr([{k:r[k] for k in ['baseline','operator','case_id','checker_receipt']} for r in S])+'\n'+script
 cmd=['python3','-S','-'] if host=='local' else [str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'),{'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[host],'python3 -S -']
 p=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=70)
 (O/(host+'_receipts.json')).write_text(p.stdout if p.returncode==0 else json.dumps({'returncode':p.returncode,'stderr':p.stderr}))
 print(host,p.returncode,len(p.stdout),p.stderr[:300],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(go,['local','server1','server2']))
(O/'HEARTBEAT.json').write_text(json.dumps({'phase':'threehost receipts captured','at':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
