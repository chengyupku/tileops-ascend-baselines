import pathlib,json,subprocess,hashlib
A=pathlib.Path(__file__).resolve().parent;D=A.parent/'transformer_attention_misc';S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
rows=json.loads((A/'SCOPE.json').read_text())
code=r'''
import sys,pathlib,json,hashlib,importlib,types,base64
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');D=W/'reports/baseline-integration-700-20261026/transformer_attention_misc'
sys.path.insert(0,str(W/'TileOPs'))
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','acl','ctypes'): raise RuntimeError('Forbidden import '+fullname)
sys.meta_path.insert(0,Block())
rows=ROWS
files={};routes=[]
def take(p):
 p=pathlib.Path(p);p=p if p.is_absolute() else W/p
 if str(p) in files:return
 try:
  b=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'content':base64.b64encode(b).decode()}
 except Exception as e:files[str(p)]={'error':str(e)}
p=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer');take(p.__file__)
for r in rows:
 mn,fn=r['factory'].replace(':','.').rsplit('.',1);m=importlib.import_module(mn);take(m.__file__)
 old=getattr(m,fn);aliases={k:v for k,v in vars(p).items() if v is old}
 stub=lambda *args:('exact',r['baseline'],r['operator'],r['case_id']);setattr(m,fn,stub)
 for k in aliases:setattr(p,k,stub)
 case=types.SimpleNamespace(id=r['case_id'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['operator'])))
 try:result=p.PROVIDERS[r['operator']](case);routes.append({'key':[r['baseline'],r['operator'],r['case_id']],'result':result,'exact':result==stub(),'formal_module':m.__file__})
 except Exception as e:routes.append({'key':[r['baseline'],r['operator'],r['case_id']],'error':repr(e)})
 setattr(m,fn,old)
 for k,v in aliases.items():setattr(p,k,v)
 for field in ('checker_receipt','guard_receipt'):take(D/r[field])
 c=json.loads((D/r['checker_receipt']).read_text())
 for field in ('source_sha256','oracle_source_sha256'):
  for path in c.get(field,{}):take(path)
for path in ('run_batch.py','run_dense.py','TEARDOWN_TIMEOUT_EVIDENCE.json','launch_dense.py'):take(D/path)
for path in ('TileOPs/scripts/ascend_pair_check.py','TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_runtime.py','reports/baseline-integration-20261026/transformer_attention/ascend_graph_call.py'):take(path)
manifest=json.loads((W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json').read_text())
cfg=manifest['groups']['attention'];pins={k:v for k,v in cfg['pins'].items() if k.endswith('libcust_opapi.so') or k.endswith('aclnn_flash_attention_score.h') or ('flash_attention_score' in k and k.endswith('.cpp'))}
critical=[]
for k,h in pins.items():
 pth=pathlib.Path(k);b=pth.read_bytes();critical.append({'path':k,'expected':h,'sha256':hashlib.sha256(b).hexdigest()})
logs=[]
for r in rows:
 run=D/'runs'/r['case_id']
 for f in run.rglob('*.log'):
  if f.stat().st_size>30000000:continue
  lines=f.read_text(errors='replace').splitlines();matches=[{'line':i+1,'text':s[:1200]} for i,s in enumerate(lines) if ('FlashAttentionScore' in s or 'flash_attention_score' in s) and ('kernel' in s.lower() or 'classicpilot' in s or 'launch' in s.lower())]
  if matches:logs.append({'case_id':r['case_id'],'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'matches':matches[:5]})
print(json.dumps({'files':files,'routes':routes,'critical':critical,'native_logs':logs,'runtime':{k:v for k,v in cfg.items() if k not in ('pins','directories')},'forbidden_imports':[k for k in sys.modules if k.split('.')[0] in ('torch','torch_npu','acl','ctypes')]}))
'''.replace('rows=ROWS','rows='+repr(rows))
p=subprocess.run([S,'openlibing-DevEnv-055409','PYTHONDONTWRITEBYTECODE=1 python3 -S -'],input=code,text=True,capture_output=True,timeout=180)
(A/'ssh_status.json').write_text(json.dumps({'exit':p.returncode,'stderr':p.stderr}));(A/'remote_evidence.json').write_text(p.stdout)
if p.returncode: print(p.stderr);raise SystemExit(p.returncode)
x=json.loads(p.stdout);import base64
for path,item in x['files'].items():
 if 'content' in item:
  dest=A/'remote_files'/path.removeprefix('/opt/tileops-all-pairs/workspace/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(base64.b64decode(item.pop('content')));item['local_copy']=str(dest)
(A/'REMOTE_INDEX.json').write_text(json.dumps(x,indent=2));(A/'remote_evidence.json').unlink()
print(json.dumps({'routes':x['routes'],'files':len(x['files']),'critical':len(x['critical']),'logs':len(x['native_logs'])}))
