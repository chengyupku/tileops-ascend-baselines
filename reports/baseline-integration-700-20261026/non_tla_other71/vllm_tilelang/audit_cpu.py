"""Read-only source/deployment audit; stdlib only, never imports baseline runtimes."""
import ast,datetime,hashlib,json,pathlib,subprocess
ROOT=pathlib.Path('/root/dyq/workspace'); OUT=pathlib.Path(__file__).parent; REPORT=ROOT/'reports/baseline-integration-700-20261026'
scope=json.loads((OUT/'SCOPE.json').read_text()); keys={tuple(x['key']) for x in scope}; prior=[]
def walk(x):
 if isinstance(x,dict):
  if isinstance(x.get('key'),list) and tuple(x['key']) in keys: prior.append(x)
  else:
   for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
for f in dict.fromkeys(x['evidence'] for x in scope):walk(json.loads((REPORT/f).read_text()))
(OUT/'PRIOR_EXACT.json').write_text(json.dumps(prior,indent=2))
files={
'mhc':'ascend_baselines/tilelang/examples/deepseek_mhc/example_mhc_post.py',
'nsa':'ascend_baselines/tilelang/examples/deepseek_nsa/example_tilelang_nsa_fwd_varlen.py',
'sigmoid':'ascend_baselines/tilelang/testing/python/math/test_math_fast_math.py',
'cumsum':'ascend_baselines/vllm-ascend/vllm_ascend/ops/triton/kda/cumsum.py',
'kda_api':'ascend_baselines/vllm-ascend/vllm_ascend/ops/triton/kda/kda.py',
'gdn_reference':'TileOPs/workloads/linear_attention/gdn.py',
'kda_reference':'TileOPs/workloads/linear_attention/kda.py',
'checker':'TileOPs/scripts/ascend_pair_check.py',
'linear_adapter':'TileOPs/benchmarks/ascend_all_pairs/vllm/linear_attention.py',
'runtime':'TileOPs/benchmarks/ascend_all_pairs/vllm/runtime.py'}
evidence={}
for name,rel in files.items():
 p=ROOT/rel; data=p.read_text();tree=ast.parse(data)
 evidence[name]={'path':str(p),'relative':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ast_parse':'passed','functions':[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno,'signature':ast.unparse(n.args)} for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]}
(OUT/'SOURCE_EVIDENCE.json').write_text(json.dumps(evidence,indent=2))
remote_script='''import pathlib,json,hashlib,platform,sys
root=pathlib.Path('/opt/tileops-all-pairs/workspace')
files=FILES
rows={}
for key,rel in files.items():
 p=root/rel
 rows[key]={'path':str(p),'exists':p.is_file(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
base=root/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'
packages=[]
for p in base.glob('*.dist-info/METADATA'):
 if any(t in p.parent.name.lower() for t in ('torch','triton','tilelang','vllm')):
  packages.append({'path':str(p),'identity':[l for l in p.read_text().splitlines() if l.startswith(('Name:','Version:'))]})
print(json.dumps({'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'host':platform.node(),'machine':platform.machine(),'python':sys.version,'sources':rows,'packages':packages,'launcher':pathlib.Path('/opt/tileops-all-pairs/bin/allpairs-python').read_text(),'bootstrap':pathlib.Path('/opt/tileops-all-pairs/control/private_python.py').read_text()},indent=2))
'''.replace('FILES',repr(files))
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','openlibing-DevEnv-661495','/usr/bin/python3 -I -S -B -']
p=subprocess.run(cmd,input=remote_script,text=True,capture_output=True,timeout=40)
(OUT/'REMOTE_DEPLOYMENT.json').write_text(json.dumps({'argv':cmd,'exit':p.returncode,'stderr':p.stderr,'data':json.loads(p.stdout) if p.returncode==0 else p.stdout},indent=2))
print(json.dumps({'scope':len(scope),'prior_exact_rows':len(prior),'source_files_ast_checked':len(evidence),'remote_exit':p.returncode}))
