import subprocess,pathlib,tarfile,io
P=pathlib.Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_attention_misc';ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code="import pathlib,tarfile,sys; p=pathlib.Path(%r); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|'); [(t.add(f,arcname=str(f.relative_to(p)),recursive=False)) for f in p.rglob('*') if f.is_file() and (('runs' in f.relative_to(p).parts and f.suffix in ('.json','.log')) or f.parent==p and f.suffix in ('.json','.py'))];t.close()"%R
r=subprocess.run([ssh,'openlibing-DevEnv-055409','python3 -'],input=code.encode(),stdout=subprocess.PIPE,check=True)
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(P,filter='data')
