import pathlib,json,hashlib,subprocess,shlex,time
P=pathlib.Path('/root/dyq/workspace/reports/baseline-flex-20261106');B=P/'batches/batch22/reference-dependency-repair';D=P/'server2';H='openlibing-DevEnv-055409';remote='/opt/baseline-flex-20261106/server2'
m=json.load(open(B/'SHA256.json'))
code='import pathlib,json,hashlib;p=pathlib.Path('+repr(remote+'/reference-python')+');m=json.load(__import__("sys").stdin);assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in m.items());(p/"SHA256.json").write_text(json.dumps(m));print(len(m))'
r=subprocess.run(['ssh','-o','ClearAllForwardings=yes',H,'python3 -c '+shlex.quote(code)],input=json.dumps(m),text=True,capture_output=True,check=True);print('TARGET_VERIFIED',r.stdout,flush=True)
while True:
 s=json.load(open(D/'STATE.json'))
 if s['phase']=='stopped' and not pathlib.Path('/proc/'+str(s['pid'])).exists():break
 time.sleep(3)
f=D/'measure_pair.py';old=f.read_bytes();(B/'measure.before.py').write_bytes(old);src=old.decode();needle="    try:\n        if task.get('manifest_overlay') == 'fixed_v010':";assert needle in src
insert="""    try:
        if op == 'DeltaNetFwdOp' and library == 'vllm-ascend':
            dependency = Path(__file__).resolve().parent / 'reference-python'
            dependency_hashes = json.loads((dependency / 'SHA256.json').read_text())
            assert all(hashlib.sha256((dependency / k).read_bytes()).hexdigest() == v for k, v in dependency_hashes.items()), 'reference dependency changed'
            sys.path.insert(0, str(dependency))
            result['reference_dependency_sha256'] = dependency_hashes
        if task.get('manifest_overlay') == 'fixed_v010':"""
src=src.replace(needle,insert);compile(src,str(f),'exec');f.write_text(src)
subprocess.run(['scp','-O','-q','-o','ClearAllForwardings=yes',str(f),H+':'+remote+'/measure_pair.py.next'],check=True)
code='import pathlib,hashlib;p=pathlib.Path('+repr(remote+'/measure_pair.py.next')+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(hashlib.sha256(f.read_bytes()).hexdigest())+';p.replace('+repr(remote+'/measure_pair.py')+')'
subprocess.run(['ssh','-o','ClearAllForwardings=yes',H,'python3 -c '+shlex.quote(code)],check=True)
retry=json.load(open(D/'RETRY.json')) if (D/'RETRY.json').exists() else {};auto=json.load(open(D/'AUTO_DEFERRED.json'));(B/'AUTO_DEFERRED.before.json').write_text(json.dumps(auto,indent=2));(B/'RETRY.before.json').write_text(json.dumps(retry,indent=2));progress=json.load(open(D/'PROGRESS.json'));now=time.time();ids=[]
for t in json.load(open(P/'batches/batch22/PLAN.json'))['rows']:
 if t['host']!='server2':continue
 assert progress.get(t['id'],{}).get('status')!='accepted'
 retry[t['id']]={'at':now,'reason':'New evidence: missing reference fla imported in canonical checker verify; isolated490file fla+einops identical SHA to passing server1, installed and verified; kernel/checker/oracle source unchanged.'};auto.pop(t['id'],None);ids.append(t['id'])
for name,data in [('RETRY.json',retry),('AUTO_DEFERRED.json',auto)]:
 f=D/name;tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(f)
# Old failure records are retained. Clear family circuit by versioning only the explicit repair gate,
# without mutating existing PLAN task records: controller gate must consider RETRY newer than failure.
f=D/'controller.py';old=f.read_bytes();(B/'controller.before.py').write_bytes(old);src=old.decode();needle="row.get('status')=='pending' and lookup.get(tid,{}).get('route_family')==family";assert needle in src
src=src.replace(needle,"row.get('status')=='pending' and read(P/'RETRY.json',{}).get(tid,{}).get('at',0) <= row.get('at',0) and lookup.get(tid,{}).get('route_family')==family");compile(src,str(f),'exec');f.write_text(src)
(D/'STOP').unlink();subprocess.run(['systemd-run','--unit=baseline-flex-server2','--property=WorkingDirectory=/root/dyq/workspace','/usr/bin/python3',str(D/'controller.py')],check=True)
(B/'DEPLOYED.json').write_text(json.dumps({'at':now,'files':len(m),'exact_retry_ids':ids,'source_host':'server1 passing canonical reference','reference_changes':False,'performance_candidate_changes':False},indent=2));print('RESUMED',ids,flush=True)
