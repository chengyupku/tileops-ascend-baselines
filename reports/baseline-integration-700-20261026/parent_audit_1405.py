import json,pathlib,hashlib,subprocess,datetime,fcntl
D=pathlib.Path(__file__).resolve().parent;A=D/'acceptance_audit01';rows=json.loads((A/'RECOMMENDED_ACCEPT.json').read_text())['rows'];(D/'PARENT_AUDIT_1405_SNAPSHOT.json').write_text(json.dumps(rows,indent=2))
S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code=r'''
import sys,pathlib,json,hashlib,fcntl,types,importlib
sys.dont_write_bytecode=True
sys.path.insert(0,'/opt/tileops-all-pairs/workspace/TileOPs')
rows=ROWS
family=FAMILY
with open('/tmp/baseline700-provider-'+family+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 p=importlib.import_module('benchmarks.ascend_all_pairs.'+family);out=[]
 for r in rows:
  op,cid=r['key'][1:];f=r['formal_factory'];factory='factory'
  if family=='tla':module=f.split(':')[0];factory=f.split(':')[-1]
  elif op=='GDNFwdOp':module='benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_linear_gdn';factory='build'
  else:
   module='benchmarks.ascend_all_pairs.ops_transformer.'+f.split('.')[0];factory=f.split('.')[-1]
  m=importlib.import_module(module);h=hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest();assert h==r['module_sha256'],(module,h,r['module_sha256'])
  stub=lambda *a,**kw:('selected',op,cid)
  old=getattr(m,factory);setattr(m,factory,stub)
  aliases={k:v for k,v in vars(p).items() if v is old}
  for k in aliases:setattr(p,k,stub)
  case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
  try:assert p.PROVIDERS[op](case)==('selected',op,cid),(op,cid)
  finally:
   setattr(m,factory,old)
   for k,v in aliases.items():setattr(p,k,v)
  out.append(dict(key=r['key'],module_sha256=h,exact_formal_dispatch=True))
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 print(json.dumps(dict(rows=out,device_imported=False)))
'''
for family,host,selected in [('tla','openlibing-DevEnv-661495',[r for r in rows if r['key'][0]=='tilelang-ascend']),('ops_transformer','openlibing-DevEnv-055409',[r for r in rows if r['key'][0]=='ops-transformer'])]:
 script=code.replace('rows=ROWS','rows='+repr(selected)).replace('family=FAMILY','family='+repr(family));r=subprocess.run([S,host,'python3 -S -'],input=script,text=True,capture_output=True,timeout=50);(D/('PARENT_AUDIT_1405_'+family+'.json')).write_text(r.stdout);assert r.returncode==0,r.stderr[-5000:];print(family,len(json.loads(r.stdout)['rows']))
for r in rows:
 for typ in ['checker','guard']:
  q=pathlib.Path(r[typ]['path']);assert hashlib.sha256(q.read_bytes()).hexdigest()==r[typ]['sha256']
 c=json.loads(pathlib.Path(r['checker']['path']).read_text());assert c['status']=='verified';v=c['verification'];assert v['checked_outputs']==v['total_outputs'] and v['total_outputs']>0;assert c['inputs_preserved'] and not c['oracle_source_changed']
 assert r['accept_recommended'] and not r['issues']
print('all',len(rows),'evidence identities and checker validated; ready parent ledger update')
