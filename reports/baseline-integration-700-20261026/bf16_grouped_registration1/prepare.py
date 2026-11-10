import ast,builtins,hashlib,json,pathlib,pprint,sys,datetime
sys.dont_write_bytecode=True
O=pathlib.Path(__file__).resolve().parent; W=O.parents[2]; B=O.parent/'fp16_grouped_registration2'; N=O.parent/'bf16_reference_budget2'
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
base=(B/'integration700_fp16_grouped_registration2.py').read_text();const={}
for node in ast.parse(base).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  try:const[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
pins=json.loads((N/'FROZEN.json').read_text())['sha256']; exclusions={}
for p in list(pins):
 if '/benchmarks/ascend_all_pairs/' in p and (p.endswith('/__init__.py') or p.endswith('/registry.py')):exclusions[p]=pins.pop(p)
for p,h in json.loads((B/'FROZEN_PINS.json').read_text()).items():
 if '/reports/' not in p:
  if p in pins:assert pins[p]==h,p
  else:pins[p]=h
execution=json.loads((N/'moe-fwd-bfloat16/result/EXECUTION.json').read_text())
loaded=execution['mapped_host_hashes']
for p,h in loaded.items():
 if p in pins:assert pins[p]==h,p
 else:pins[p]=h
local=const['LOCAL_FILES'][:];local[4]=str(N/'adapter.py')
for p in local:
 if p not in pins:pins[p]=sha(p)
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
plan=entry_plan('GroupedGemmFwdOp',load_manifest()['GroupedGemmFwdOp'],load_adts(),resolve=False)
calls=[instantiate(plan,r,d) for r in load_workloads('GroupedGemmFwdOp') for d in r.get('dtype_cases') or [{}]]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
params={};indices={};contracts={};envelopes={}
rows=[r for r in json.loads((N/'CASES.json').read_text()) if r['key']==['ops-transformer','GroupedGemmFwdOp','moe-fwd-bfloat16']];assert len(rows)==1
for row in rows:
 key=tuple(row['key']);c=next(c for c in calls if c.case_id==key[2]);receipt=json.loads((N/key[2]/'result/representative.json').read_text())
 assert canonical(dict(c.tensors,**c.params))==canonical(receipt['case_snapshot']['parameters'])
 assert c.indices==row['manifest_params']
 params[key]=canonical(receipt['case_snapshot']['parameters']);indices[key]=canonical(c.indices)
 contracts[key]=canonical(dict(indices=c.indices,rawparams=c.params,tensors=c.tensors,specs={n:None if s is None else dict(shape=s.shape,dtype=s.dtype,values=s.values,cpu=s.cpu) for n,s in c.specs.items()},presence={n:c.present(n) for n in set(c.specs)|{'out'}},has_out=c.has_out))
 envelopes[key[2]]=dict(row['envelope'],params=receipt['case_snapshot']['parameters'])
 for p in (N/key[2]/'result').glob('*.json'):pins[str(p)]=sha(p)
for p in [N/'PARENT_REVIEW.json',N/'FROZEN.json',N/'CASES.json']:pins[str(p)]=sha(p)
dump('FROZEN_PINS.json',pins)
dump('PIN_PROVENANCE.json',dict(original_frozen=str(N/'FROZEN.json'),original_hash=sha(N/'FROZEN.json'),excluded_publication_routes=exclusions,template=str(B/'integration700_fp16_grouped_registration2.py'),template_sha256=sha(B/'integration700_fp16_grouped_registration2.py'),bootstrap_source=str(B/'FROZEN_PINS.json'),execution_source=str(N/'moe-fwd-bfloat16/result/EXECUTION.json'),original_math='BF16 Grouped adapter + bf16_local_gemm18/binding.py; unchanged',pins=len(pins)))
env=const['ENV'];historical=json.loads((N/'moe-fwd-bfloat16/result/RUNTIME.json').read_text())['environment']
assert all(env[k]==v for k,v in historical.items() if k in env)
head='"""Exact reviewed BF16 Grouped NT registration; pending parent publication."""\nimport builtins, hashlib, json, os, types, tempfile, shutil, sys, ast\nfrom pathlib import Path\nfrom types import MappingProxyType\n'
for name,val in [('ROOT',const['ROOT']),('MANIFEST',str(O/'FROZEN_PINS.json')),('MANIFEST_HASH',sha(O/'FROZEN_PINS.json')),('LOCAL_FILES',local),('CASE_PARAMS',params),('CASE_INDICES',indices),('CALL_CONTRACTS',contracts),('ENVELOPES',envelopes),('ENV',env),('LOADED_VENDOR',loaded),('REQUIRED_LOADED',const['REQUIRED_LOADED'])]:head+=name+' = '+pprint.pformat(val,width=120)+'\n'
body=base[base.index('ROOT = Path(ROOT)'):].replace('FP16','BF16').replace('fp16_grouped','bf16_grouped').replace('fp16-grouped-registration2-','bf16-grouped-registration1-').replace('Exact two-key','Exact one-key').replace('grouped NN/NT cases.','grouped NT case.').replace('bf16_grouped_budget900_2','bf16_reference_budget2')
(O/'integration700_bf16_grouped_registration1.py').write_text(head+'\n'+body)
for name in ('run_environment.sh','run_case.sh'):
 text=(B/name).read_text().replace('FP16','BF16').replace('fp16_grouped_registration2','bf16_grouped_registration1').replace('fp16_grouped_budget900_2','bf16_reference_budget2').replace('moe-dgrad-float16|moe-fwd-float16','moe-fwd-bfloat16')
 (O/name).write_text(text)
dump('CONTRACTS.json',dict(status='prepared_pending_parent_review',accepted_added=0,rows=rows,contracts={k[2]:json.loads(v) for k,v in contracts.items()},envelopes=envelopes,review=json.loads((N/'PARENT_REVIEW.json').read_text()),pins=len(pins)))
with (O/'FEEDBACK.txt').open('a') as f:f.write('Prepared one BF16 NT contract using published FP16 framework, original BF16 adapter and actual EXECUTION maps. No NPU; full pin hash deferred to CPU checks once.\n')
print('prepared',len(pins),'pins')
