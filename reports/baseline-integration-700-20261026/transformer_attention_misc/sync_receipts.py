import subprocess,pathlib,tarfile,io
P=pathlib.Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_attention_misc';ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code="""import pathlib,tarfile,sys
p=pathlib.Path(%r); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')
files=list((p/'runs').glob('*/*.json'))+list((p/'runs').glob('*/guard.log'))+list(p.glob('*COMPILE*.json'))+list(p.glob('grouped*.log'))+list(p.glob('REGISTRATION*.json'))+list(p.glob('provider*.patch'))+list(p.glob('init-before-*.py'))
files += [p/n for n in ['TEARDOWN_TIMEOUT_EVIDENCE.json','ACTUAL_SOURCE_HASHES.json','ATTENTION_RUNTIME_MANIFEST.json','RERUN_REASON.json','REGISTRATION_CHECK.json','grouped_bridge.so']]
files += list(p.glob('ops_tensor-init-before-*.py'))+list((p/'vllm_private').glob('*'))+list((p/'prior_deepep').glob('*'))
for f in files:
 if f.is_file():t.add(f,arcname=str(f.relative_to(p)),recursive=False)
t.close()
"""%R
r=subprocess.run([ssh,'openlibing-DevEnv-055409','python3 -'],input=code.encode(),stdout=subprocess.PIPE,check=True)
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(P,filter='data')
