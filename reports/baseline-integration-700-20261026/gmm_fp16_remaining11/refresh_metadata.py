from pathlib import Path
import subprocess,json
P=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11'
code='''from pathlib import Path
import json
p=Path(ROOT);result={}
for d in [p]+[q for q in p.iterdir() if q.is_dir() and (q/'capture.py').exists()]:
 for n in ['REAL_REGISTRY_CPU_PROOF.json','REGISTRATION_PROOF.json','representative.json','EXECUTION.json','guard.exit','CAPTURE.json','guard_snapshot.json','registered_function.py','registration_append.py','registered_module.py']:
  f=d/n
  if f.exists():result[str(f.relative_to(p))]=f.read_text()
root=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer')
result['final_provider_module.py']=(root/'integration700_gmm_fp16_remaining11.py').read_text()
result['final_registry.py']=(root/'__init__.py').read_text()
print(json.dumps(result))
'''.replace('ROOT',repr(r))
x=subprocess.run([ssh,'openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True);x.check_returncode()
for n,s in json.loads(x.stdout).items():
 f=P/n;f.parent.mkdir(exist_ok=True);f.write_text(s)
print('refreshed',len(json.loads(x.stdout)),'metadata files')
