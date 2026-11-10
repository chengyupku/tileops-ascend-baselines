import pathlib,subprocess,json
P=pathlib.Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code='''import pathlib,json,hashlib
W=pathlib.Path('/opt/tileops-all-pairs/workspace');R=W/'reports/baseline-integration-700-20261026/sgl_attention_recovery32'
paths=[W/'TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py',W/'TileOPs/benchmarks/ascend_all_pairs/sgl/integration700_sgl_gdn_register30.py',W/'TileOPs/benchmarks/ascend_all_pairs/sgl/integration700_sgl_attention_register10.py']
paths += [R/n for n in ['provider_init.before.py','provider_init.after_race.py','dsa_adapter.py','runs/dsa07_11/checker.json','runs/dsa07_11/LOADED_MODULES.json','runs/dsa07_11/NATIVE_ARTIFACTS.json']]
paths += [pathlib.Path(x['path']) for x in json.loads((R/'SOURCE_HASHES.json').read_text()) if '/attention/decode_attention.py' in x['path'] or x['path'].endswith('/__init__.py') or x['path'].endswith('.so')]
c=json.loads((R/'runs/dsa07_11/checker.json').read_text());paths += [W/k for k in c['oracle_source_sha256']]
result=[]
for p in paths:
 if not p.exists():
  result.append({'path':str(p),'missing':True});continue
 b=p.read_bytes(); result.append({'path':str(p),'resolved':str(p.resolve()),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode() if p.suffix=='.py' and '/sgl/' in str(p) else None})
print(json.dumps(result))
'''
r=subprocess.run([SSH,'openlibing-DevEnv-055409','python3 -'],input=code,text=True,capture_output=True,timeout=60)
(P/'REMOTE_READ.json').write_text(r.stdout);(P/'REMOTE_READ.stderr').write_text(r.stderr);print('rc',r.returncode)
if r.returncode:print(r.stderr)
else:
 for x in json.loads(r.stdout):print(x['path'],x.get('sha256','MISSING'))
