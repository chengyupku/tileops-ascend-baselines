import pathlib,json,subprocess
D=pathlib.Path(__file__).resolve().parent
scope=json.loads((D/'SCOPE.json').read_text())
code=r'''
import pathlib,json,hashlib,sys,importlib,types,ast,subprocess
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/tla_simple_tilelang'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(s):
 p=pathlib.Path(s.replace('/root/dyq/workspace',str(W)))
 if p.is_absolute():return p
 return next((x for x in [P/p,W/p] if x.exists()),W/p)
def ref(p):return {'path':str(p),'sha256':sha(p)}
sys.path.insert(0,str(W/'TileOPs'))
# Guard imports before loading real registry: no device library can be loaded.
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','torch_npu','tilelang','tvm','ctypes'}:raise RuntimeError('Forbidden device import '+name)
sys.meta_path.insert(0,Block())
p=importlib.import_module('benchmarks.ascend_all_pairs.tla')
registry=importlib.import_module('benchmarks.ascend_all_pairs.registry')
m=importlib.import_module('benchmarks.ascend_all_pairs.tla.integration700_tla_simple_tilelang_gemm_rest7')
m.inspect_pins()
m._prepare=lambda:(None,None,types.SimpleNamespace(factory=lambda c,op,n,t,o:dict(key=['tilelang-ascend',op,c.id],params=c.params,device_construction_stub=True)),None)
files={};rows=[]
def capture(path,content=False):
 path=resolve(str(path));r=ref(path)
 if content:r['text']=path.read_text()
 files[str(path)]=r;return r
for s in ['gemm_rest_pto.guard.json','gemm_rest_pto.guard.log','gemm_rest_pto.json','run_batch9.py','launch_batch9.py','gemm_adapter_pto.py','frozen_gemm_rest7/adapter.py','frozen_gemm_rest7/private_tla/native.py','frozen_gemm_rest7/private_tla/deployment.py']:
 capture(P/s,True)
for s in ['TileOPs/scripts/ascend_pair_check.py','TileOPs/workloads/gemm.py','TileOPs/workloads/numerics.py','TileOPs/benchmarks/ascend_all_pairs/jit_evidence.py','tools/baseline_protocol.py']:
 capture(W/s,True)
capture(pathlib.Path(m.__file__),True);capture(pathlib.Path(p.__file__),True);capture(pathlib.Path(registry.__file__),True)
for rel in ['tilelang/jit/__init__.py','tilelang/utils/target.py','tilelang/engine/lower.py']:
 capture(pathlib.Path(m.config()['runtime'])/rel,True)
for rel in ['TileOPs/src/tileops/manifest/spec/gemm.yaml','TileOPs/src/tileops/manifest/spec/gemm_bmm.yaml','TileOPs/workloads/workload_base.py']:
 capture(W/rel,True)
for s in ['examples/gemm/example_gemm.py','examples/batch_gemm/batch_gemm.py']:
 capture(W/'ascend_baselines/tilelang-ascend'/s,True)
for r in SCOPE:
 cp=resolve(r['checker_receipt']);gp=resolve(r['guard_receipt']);c=json.loads(cp.read_text());g=json.loads(gp.read_text());issues=[];hs={};calls=[];kernels=[]
 for s,h in {**c['source_sha256'],**c['oracle_source_sha256'],**m.config()['pins']}.items():
  fp=resolve(s)
  try:
   actual=sha(fp);hs[str(fp)]={'expected':h,'actual':actual,'matches':h==actual}
   if h!=actual:issues.append('hash mismatch '+str(fp))
   if fp.name.startswith('calls-'):
    calls.append(dict(**ref(fp),data=json.loads(fp.read_text())))
   if fp.name.endswith('.kernel.txt'):kernels.append(capture(fp,True))
  except OSError as e:issues.append(str(e))
 case=types.SimpleNamespace(id=r['case_id'],params=c['case_snapshot']['parameters'])
 case._call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['operator']))
 route=registry.candidate_factory(case,r['baseline'])()
 assert route['key']==[r['baseline'],r['operator'],r['case_id']]
 bad=types.SimpleNamespace(id=case.id,params={})
 try:p.PROVIDERS[r['operator']](bad);raise AssertionError('bad params accepted')
 except ValueError:pass
 rows.append(dict(key=route['key'],checker=dict(**ref(cp),data=c),child_guard=dict(**ref(gp),data=g),hashes=hs,calls=calls,kernels=[{k:v for k,v in x.items() if k!='text'} for x in kernels],route=route,issues=issues))
assert not any(n in sys.modules for n in ['torch','torch_npu','tilelang','tvm','ctypes'])
print(json.dumps(dict(rows=rows,files=files,formal_module=ref(pathlib.Path(m.__file__)),provider=ref(pathlib.Path(p.__file__)),device_imported=False)))
'''.replace('for r in SCOPE:','for r in '+repr(scope)+':')
(D/'remote_check.py').write_text(code)
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r=subprocess.run([ssh,'openlibing-DevEnv-661495','PYTHONDONTWRITEBYTECODE=1 python3 -S -'],input=code,text=True,capture_output=True,timeout=100)
(D/'SSH_RECEIPT.json').write_text(json.dumps({'command':[ssh,'openlibing-DevEnv-661495','PYTHONDONTWRITEBYTECODE=1 python3 -S -'],'returncode':r.returncode,'stderr':r.stderr},indent=2))
(D/'REMOTE_EVIDENCE.json').write_text(r.stdout)
print('exit',r.returncode,'bytes',len(r.stdout),'stderr',r.stderr[-3000:])
