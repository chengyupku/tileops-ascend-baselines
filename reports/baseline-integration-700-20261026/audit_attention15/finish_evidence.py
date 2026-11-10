import pathlib,json,subprocess
A=pathlib.Path(__file__).resolve().parent;S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';rows=json.loads((A/'SCOPE.json').read_text())
code=r'''
import pathlib,json,hashlib,sys,types,importlib
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');D=W/'reports/baseline-integration-700-20261026/transformer_attention_misc';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();out={'files':{},'critical':[]};rows=ROWS
def take(p):
 p=pathlib.Path(p)
 if p.exists():out['files'][str(p)]={'sha256':sha(p),'text':p.read_text()}
for r in rows:
 for p in [(D/r['guard_receipt']).with_name('guard.log'),(D/r['guard_receipt']).with_name('command.json'),D/'runs'/r['case_id']/'started.json']:take(p)
for p in ['run_stable.py','launch_stable.py','dsa_small_batch.json','mla_large_batch.json']:take(D/p)
m=json.loads((W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json').read_text())
for p,h in m['groups']['attention']['pins'].items():
 if ('flash_attention_score' in p and ('/op_kernel/' in p or '/dynamic/' in p)) or p.endswith('liboptiling.so'):
  out['critical'].append({'path':p,'expected':h,'sha256':sha(p)})
# Read-only GDN live routing and source equivalence. No device constructors.
sys.path.insert(0,str(W/'TileOPs'))
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','ctypes','acl'):raise RuntimeError('blocked '+fullname)
sys.meta_path.insert(0,Block())
p=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer');g=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_linear_gdn');take(g.__file__);take(g.PRIVATE/'REGISTERED.json')
cid='q35-397b-tp8-1024-7168-bfloat16';case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GDNFwdOp')))
g.build=lambda case:('exact_gdn',case.id)
try:out['gdn_current_route']=p.PROVIDERS['GDNFwdOp'](case)
except Exception as e:out['gdn_current_route']={'error':str(e)}
for name in ['binding.py','entry.py','direct_acl.py']:
 a=g.PRIVATE/'chunk_package'/name;b=g.PRIVATE/'frozen_v1/chunk_package'/name
 out.setdefault('gdn_frozen_equivalence',[]).append({'tested_path':str(a),'tested_sha256':sha(a),'registered_path':str(b),'registered_sha256':sha(b),'equal':a.read_bytes()==b.read_bytes()})
out['device_imports']=[k for k in sys.modules if k.split('.')[0] in ('torch','torch_npu','ctypes','acl')]
print(json.dumps(out))
'''.replace('ROWS',repr(rows))
p=subprocess.run([S,'openlibing-DevEnv-055409','PYTHONDONTWRITEBYTECODE=1 python3 -S -'],input=code,text=True,capture_output=True,timeout=120);(A/'FINAL_REMOTE.json').write_text(p.stdout);(A/'final_ssh_status.json').write_text(json.dumps({'exit':p.returncode,'stderr':p.stderr}));print(p.returncode,p.stderr)
