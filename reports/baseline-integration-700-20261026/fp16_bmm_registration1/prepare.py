import ast,hashlib,json,pathlib,pprint,sys,builtins
O=pathlib.Path(__file__).resolve().parent; W=pathlib.Path('/root/dyq/workspace'); B=O.parent/'bf16_bmm_registration7'; N=O.parent/'fp16_local_remaining3'; C=N/'square-2k-float16'
sys.path[:0]=[str(W/'TileOPs/src'),str(W/'TileOPs'),str(W/'.venv/lib/python3.12/site-packages')]
original=builtins.__import__
def safe(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl'):raise AssertionError(name)
 return original(name,*a,**kw)
builtins.__import__=safe
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(n,d):(O/n).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
base=(B/'integration700_bf16_bmm_registration7.py').read_text(); const={}
for node in ast.parse(base).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  try:const[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
pins=json.loads((C/'FROZEN.json').read_text())['sha256']; after=json.loads((C/'PINS_AFTER.json').read_text());assert all(after[p]==h for p,h in pins.items())
excluded={}
for p in list(pins):
 if p.endswith('/ops_transformer/__init__.py'):excluded[p]={'historical_sha256':pins.pop(p),'reason':'Shared provider publication routing only'}
local=const['LOCAL_FILES'][:];local[4]=str(N/'adapter.py')
for p in local:
 assert p in pins or p==local[0] or p==local[5],p
 if p in pins:assert sha(p)==pins[p],p
 else:pins[p]=sha(p)
for p in (W/'TileOPs/src/tileops/manifest').rglob('*'):
 if p.is_file() and p.suffix in ('.py','.yaml'):pins[str(p)]=sha(p)
review=json.loads((N/'PARENT_BMM_PARTIAL_REVIEW.json').read_text());r=C/'result/representative.json';assert sha(r)==review['checker_sha256']; receipt=json.loads(r.read_text());assert receipt['input_seed']==1409789906
ad=json.loads((C/'result/RESOURCE_ADMISSION.json').read_text());assert ad['input_elements']==67108864 and ad['output_elements']==33554432
assert json.loads((C/'GUARD_FINAL.json').read_text())['exit_code']==0
for p in [C/'FROZEN.json',C/'PINS_AFTER.json',C/'GUARD_FINAL.json',N/'PARENT_BMM_PARTIAL_REVIEW.json',N/'BMM_DELIVERY.json']+list((C/'result').glob('*.json')):pins[str(p)]=sha(p)
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
plan=entry_plan('BmmFwdOp',load_manifest()['BmmFwdOp'],load_adts(),resolve=False)
calls=[instantiate(plan,row,dt) for row in load_workloads('BmmFwdOp') for dt in row.get('dtype_cases') or [{}]]
call=next(c for c in calls if c.case_id=='square-2k-float16');key=('ops-transformer','BmmFwdOp',call.case_id)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
payload={'indices':call.indices,'rawparams':call.params,'tensors':call.tensors,'specs':{n:None if s is None else {'shape':s.shape,'dtype':s.dtype,'values':s.values,'cpu':s.cpu} for n,s in call.specs.items()},'presence':{n:call.present(n) for n in set(call.specs)|{'out'}},'has_out':call.has_out}
params=receipt['case_snapshot']['parameters']; assert canonical(call.tensors)==canonical(params)==canonical(ad['params']); assert call.indices=={'B':8,'M':2048,'N':2048,'K':2048,'T':'float16'}
envelope={n:ad[n] for n in ('input_elements','output_elements','host_required','device_required','disk_required','workspace_allowance','params')}
# Actual startup hook prepends its own vendor: reproduce its passive environment effect.
env=const['ENV'].copy();pth=W/'.venv/lib/python3.12/site-packages/fla_npu_opp_env.py'
import os
saved=dict(os.environ)
try:
 os.environ.update(env);exec(compile(pth.read_bytes(),str(pth),'exec'),{'__file__':str(pth)});env['ASCEND_CUSTOM_OPP_PATH']=os.environ['ASCEND_CUSTOM_OPP_PATH']
finally:os.environ.clear();os.environ.update(saved)
for p,h in json.loads((B/'FROZEN_PINS.json').read_text()).items():
 if '/reports/' not in p:
  if p in pins:assert pins[p]==h,p
  else:pins[p]=h
assert str(pth) in pins
for p,h in pins.items():assert sha(p)==h,p
dump('FROZEN_PINS.json',pins);dump('PIN_EXCLUSIONS.json',excluded)
head='"""Exact reviewed FP16 Bmm registration; publication remains parent-owned."""\nimport builtins, hashlib, json, os, types, tempfile, shutil, sys, ast\nfrom pathlib import Path\nfrom types import MappingProxyType\n'
for name,val in [('ROOT',const['ROOT']),('MANIFEST',str(O/'FROZEN_PINS.json')),('MANIFEST_HASH',sha(O/'FROZEN_PINS.json')),('LOCAL_FILES',local),('CASE_PARAMS',{key:canonical(params)}),('CASE_INDICES',{key:canonical(call.indices)}),('CALL_CONTRACT',canonical(payload)),('ENVELOPES',{key[2]:envelope}),('ENV',env),('LOADED_VENDOR',const['LOADED_VENDOR']),('REQUIRED_LOADED',const['REQUIRED_LOADED'])]:head+=name+' = '+pprint.pformat(val,width=120)+'\n'
body=base[base.index('ROOT = Path(ROOT)'):]
start=body.index('def _canonical(');end=body.index('\ndef _proc(')
body=body[:start]+(O/'private_body.txt').read_text()+'\n'+body[end:]
body=body.replace('admission = dict(envelope,host=host', 'admission = dict(_plain(envelope),host=host')
body=body.replace('BF16_BMM','FP16_BMM').replace('bf16-bmm-registration7-','fp16-bmm-registration1-').replace('Frozen BF16 Bmm','Frozen FP16 Bmm').replace('Exact seven-key BF16','Exact one-key FP16').replace('exact six BF16 Bmm and one Grouped case.','exact FP16 square-2k Bmm case.').replace('No exact BF16 Bmm/Grouped route','No exact FP16 Bmm route')
body=body.replace("ROOT.parent/'bf16_bmm_conv_grouped_native10', ROOT.parent/'bf16_bmm_conv_grouped_prep10'","ROOT.parent/'fp16_local_remaining3'")
(O/'integration700_fp16_bmm_registration1.py').write_text(head+'\n'+body)
recipe=(B/'run_environment.sh').read_text().replace('BF16_BMM','FP16_BMM').replace('bf16_bmm_conv_grouped_native10','fp16_local_remaining3')
(O/'run_environment.sh').write_text(recipe)
dump('CONTRACTS.json',{'status':'prepared_pending_parent_publication','accepted_added':0,'key':key,'params':params,'call':json.loads(canonical(payload)),'envelope':envelope,'review':review,'pins':len(pins),'exclusions':excluded,'startup_environment':env,'implementation':'Unchanged verified adapter.candidate -> make_runner -> original binding.dense_fp32; one TensorList full batch 8; lossless FP16 to FP32 widening and original dtype cast','admission':'Live guard constructor before inputs; consume token once without repeating full peak envelope after reference'})
print('generated',len(pins),'pins')
