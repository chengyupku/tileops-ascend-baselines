import pathlib,json,ast,hashlib,subprocess,shlex
O=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/acceptance_audit01');W='/opt/tileops-all-pairs/workspace';S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code='''import pathlib,hashlib,json,ast
W=pathlib.Path('/opt/tileops-all-pairs/workspace'); out={}
for family in ['tla','ops_transformer']:
 p=W/'TileOPs/benchmarks/ascend_all_pairs'/family/'__init__.py';out[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
for p in (W/'TileOPs/benchmarks/ascend_all_pairs/tla').glob('integration700*element*.py'):
 for f in ast.parse(p.read_text()).body:
  if isinstance(f,ast.FunctionDef) and f.name=='config':
   c=ast.literal_eval(f.body[0].value)
   for s,h in c['pins'].items():
    q=pathlib.Path(s);out[s]={'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'expected':h}
for p in (W/'TileOPs/benchmarks/ascend_all_pairs/tla').glob('integration700*select3.py'):
 for f in ast.parse(p.read_text()).body:
  if isinstance(f,ast.FunctionDef) and f.name=='config':
   c=ast.literal_eval(f.body[0].value)
   for s,h in c['pins'].items():
    q=pathlib.Path(s);out[s]={'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'expected':h}
p=W/'reports/baseline-integration-700-20261026/transformer_linear/REGISTERED.json'
if p.exists():out[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
print(json.dumps(out))
'''
for h,target in [('server1','openlibing-DevEnv-661495'),('server2','openlibing-DevEnv-055409')]:
 r=subprocess.run([S,target,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=60);(O/(h+'_static.json')).write_text(r.stdout);print(h,r.returncode,r.stderr[-300:])
