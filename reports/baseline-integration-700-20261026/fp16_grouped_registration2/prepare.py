import ast,builtins,hashlib,json,pathlib,pprint,sys,datetime
O=pathlib.Path(__file__).resolve().parent; W=O.parents[2]; B=O.parent/'fp16_bmm_registration1'; N=O.parent/'fp16_grouped_budget900_2'
def dump(n,d):(O/n).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
dump('HEARTBEAT.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'prepare','npu':False})
sys.path[:0]=[str(W/'TileOPs/src'),str(W/'TileOPs'),str(W/'.venv/lib/python3.12/site-packages')]
orig=builtins.__import__
def safe(n,*a,**kw):
 if n.split('.')[0] in ('torch','torch_npu','acl'):raise AssertionError(n)
 return orig(n,*a,**kw)
builtins.__import__=safe
base=(B/'integration700_fp16_bmm_registration1.py').read_text();const={}
for node in ast.parse(base).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  try:const[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
pins=json.loads((N/'FROZEN.json').read_text())['sha256']; exclusions={}
for p in list(pins):
 if '/benchmarks/ascend_all_pairs/' in p and (p.endswith('/__init__.py') or p.endswith('/registry.py')):
  exclusions[p]=pins.pop(p)
for p,h in json.loads((B/'FROZEN_PINS.json').read_text()).items():
 if '/reports/' not in p:
  if p in pins:assert pins[p]==h,p
  else:pins[p]=h
local=const['LOCAL_FILES'][:];local[4]=str(N/'adapter.py')
for p in local:
 if p not in pins:pins[p]=sha(p)
for p in (W/'TileOPs/src/tileops/manifest').rglob('*'):
 if p.is_file() and p.suffix in ('.py','.yaml'):pins.setdefault(str(p),sha(p))
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
plan=entry_plan('GroupedGemmFwdOp',load_manifest()['GroupedGemmFwdOp'],load_adts(),resolve=False)
calls=[instantiate(plan,r,d) for r in load_workloads('GroupedGemmFwdOp') for d in r.get('dtype_cases') or [{}]]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
params={};indices={};contracts={};envelopes={};rows=json.loads((N/'CASES.json').read_text())
for row in rows:
 key=tuple(row['key']);c=next(c for c in calls if c.case_id==key[2]); receipt=json.loads((N/key[2]/'result/representative.json').read_text())
 assert canonical(dict(c.tensors,**c.params))==canonical(receipt['case_snapshot']['parameters'])
 assert c.indices==row['manifest_params']
 params[key]=canonical(receipt['case_snapshot']['parameters']);indices[key]=canonical(c.indices)
 contracts[key]=canonical(dict(indices=c.indices,rawparams=c.params,tensors=c.tensors,specs={n:None if s is None else dict(shape=s.shape,dtype=s.dtype,values=s.values,cpu=s.cpu) for n,s in c.specs.items()},presence={n:c.present(n) for n in set(c.specs)|{'out'}},has_out=c.has_out))
 envelopes[key[2]]=dict(row['envelope'],params=receipt['case_snapshot']['parameters'])
 for p in (N/key[2]/'result').glob('*.json'):pins[str(p)]=sha(p)
for p in [N/'PARENT_REVIEW.json',N/'FROZEN.json',N/'CASES.json']:pins[str(p)]=sha(p)
for p,h in pins.items():assert sha(p)==h,p
dump('FROZEN_PINS.json',pins)
dump('PIN_PROVENANCE.json',dict(original_frozen=str(N/'FROZEN.json'),original_hash=sha(N/'FROZEN.json'),excluded_publication_routes=exclusions,template=str(B/'integration700_fp16_bmm_registration1.py'),template_sha256=sha(B/'integration700_fp16_bmm_registration1.py'),bootstrap_source=str(B/'FROZEN_PINS.json'),original_math='GroupedGemm adapter + bf16_local_gemm18/binding.py; unchanged',pins=len(pins)))
head='"""Reviewed exact FP16 GroupedGemm registration. Parent-owned publication."""\nimport builtins, hashlib, json, os, types, tempfile, shutil, sys, ast\nfrom pathlib import Path\nfrom types import MappingProxyType\n'
for name,val in [('ROOT',const['ROOT']),('MANIFEST',str(O/'FROZEN_PINS.json')),('MANIFEST_HASH',sha(O/'FROZEN_PINS.json')),('LOCAL_FILES',local),('CASE_PARAMS',params),('CASE_INDICES',indices),('CALL_CONTRACTS',contracts),('ENVELOPES',envelopes),('ENV',const['ENV']),('LOADED_VENDOR',const['LOADED_VENDOR']),('REQUIRED_LOADED',const['REQUIRED_LOADED'])]:head+=name+' = '+pprint.pformat(val,width=120)+'\n'
body=base[base.index('ROOT = Path(ROOT)'):].replace('==CALL_CONTRACT)', '==CALL_CONTRACTS.get(_key(case,baseline)))')
body=body.replace('FP16_BMM','FP16_GROUPED').replace('fp16_bmm_verified','fp16_grouped_verified').replace('fp16-bmm-registration1-','fp16-grouped-registration2-').replace('FP16 Bmm','FP16 GroupedGemm').replace('Exact one-key FP16','Exact two-key FP16').replace('exact FP16 square-2k Bmm case.','exact FP16 grouped NN/NT cases.').replace("ROOT.parent/'fp16_local_remaining3'","ROOT.parent/'fp16_grouped_budget900_2'")
body=body.replace('CASE_INDICES = MappingProxyType(CASE_INDICES)','CALL_CONTRACTS = MappingProxyType(CALL_CONTRACTS)\nCASE_INDICES = MappingProxyType(CASE_INDICES)')
(O/'integration700_fp16_grouped_registration2.py').write_text(head+'\n'+body)
(O/'run_environment.sh').write_text((B/'run_environment.sh').read_text().replace('FP16_BMM','FP16_GROUPED').replace('fp16_local_remaining3','fp16_grouped_budget900_2'))
dump('CONTRACTS.json',dict(status='prepared_pending_parent_review',accepted_added=0,rows=rows,contracts={k[2]:json.loads(v) for k,v in contracts.items()},envelopes=envelopes,review=json.loads((N/'PARENT_REVIEW.json').read_text()),pins=len(pins)))
with (O/'FEEDBACK.txt').open('a') as f:f.write('Prepared exact two-key contracts, unchanged original Grouped adapter/binding and full template security framework.\n')
print('prepared',len(pins),'pins')
