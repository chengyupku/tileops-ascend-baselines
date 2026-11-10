import pathlib,json,hashlib,subprocess,shlex,datetime
ROOT=pathlib.Path('/root/dyq/workspace');OUT=ROOT/'reports/baseline-integration-700-20261026/acceptance_audit01';BASE=OUT.parent
scope=json.loads((OUT/'SCOPE.json').read_text())
def local(p):return pathlib.Path(str(p).replace('/opt/tileops-all-pairs/workspace',str(ROOT)))
def path(r,t):
 p=r['result'][t];p=p['path'] if isinstance(p,dict) else p;p=local(p);return p if p.is_absolute() else pathlib.Path(r['directory'])/p
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]; requests={h:set() for h in ['server1','server2']}
for r in scope:
 c=path(r,'checker_receipt');g=path(r,'guard_receipt');j=json.loads(c.read_text());row={'key':[r['baseline'],r['operator'],r['case_id']],'worker':r['worker'],'host':r['host'],'checker':{'path':str(c),'sha256':sha(c)},'guard':{'path':str(g),'sha256':sha(g)},'checker_summary':{k:v for k,v in j.items() if k not in ['observed_calls','case_snapshot','source_sha256','provider_source_sha256','oracle_source_sha256']},'source':j.get('source_sha256',{}),'oracle':j.get('oracle_source_sha256',{}),'provider':j.get('provider_source_sha256',{})}
 row['inputs_metadata_equal']=j['observed_calls']['reference'][0]['inputs']==j['observed_calls']['candidate'][0]['inputs']
 if r['host']!='local':
  requests[r['host']].update(str(p).replace(str(ROOT),'/opt/tileops-all-pairs/workspace') for p in [c,g])
 rows.append(row)
for host in requests:
 for family in (['tla','ops_transformer'] if host=='server1' else ['ops_transformer']):
  requests[host].add('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/'+family+'/__init__.py')
 for r,row in zip(scope,rows):
  if r['host']!=host:continue
  factory=r['result']['factory'].replace(':','.').split('.')
  if factory[0]=='benchmarks':mp='TileOPs/'+('/'.join(factory[:-1]))+'.py'
  else:mp='TileOPs/benchmarks/ascend_all_pairs/ops_transformer/'+factory[0]+'.py'
  requests[host].add('/opt/tileops-all-pairs/workspace/'+mp)
  for p in row['source']:
   if ('integration700_' in p or 'frozen_v1' in p or p.endswith('binding.py') or p.endswith('direct_acl.py') or p.endswith('libcust_opapi.so') or p.endswith('libnnopbase.so')):
    requests[host].add(str(p) if str(p).startswith('/opt/') else '/opt/tileops-all-pairs/workspace/'+str(p))
script='''import pathlib,json,hashlib
paths=json.loads(INPUT)
out={}
for s in paths:
 p=pathlib.Path(s)
 try:out[s]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size}
 except Exception as e:out[s]={'error':str(e)}
print(json.dumps(out))
'''
ssh=str(ROOT/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh')
for host,paths in requests.items():
 code=script.replace('INPUT',repr(json.dumps(sorted(paths))))
 res=subprocess.run([ssh,{'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[host],'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=90)
 (OUT/(host+'_hashes.json')).write_text(res.stdout);print(host,res.returncode,len(paths),res.stderr[-300:])
(OUT/'EVIDENCE_INDEX.json').write_text(json.dumps(rows,indent=2))
print('collected',len(rows))
