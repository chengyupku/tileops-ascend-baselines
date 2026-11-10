import subprocess,tarfile,json,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11'
import sys
sub=sys.argv[1] if len(sys.argv)>1 else ''
target=r+('/'+sub if sub else '')
code='''from pathlib import Path
import json,hashlib,gzip
p=Path(TARGET)
s=p/'source_snapshot';s.mkdir(exist_ok=True)
m=json.loads((p/'FROZEN.json').read_text())['sha256']
for f in m:
 q=Path(f)
 if q.suffix in ('.py','.h','.yaml') and ('/TileOPs/' in f or q.name in ('integration700_transformer_dense_gmm.py','aclnn_grouped_matmul_v5.h')):
  (s/q.name).write_bytes(q.read_bytes())
logs=list((p/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'))
lines=[]
for q in logs:
 text=q.read_text(errors='replace')
 for line in text.splitlines():
  if any(t in line for t in ['GroupedMatmul','grouped_matmul','Load binary','load binary','private_provider']):lines.append(line)
 with gzip.open(p/(q.name+'.gz'),'wb') as z:z.write(q.read_bytes())
(p/'native_dispatch_excerpt.log').write_text('\\n'.join(lines)+'\\n')
files=[q for q in p.iterdir() if q.is_file() and q.name not in ('REMOTE_STATIC.json',)]
(p/'REMOTE_ARTIFACTS.json').write_text(json.dumps({q.name:{'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size} for q in files},indent=2))
print(json.dumps([q.name for q in p.iterdir() if q.is_file()]+['source_snapshot']))
'''.replace('TARGET',repr(target))
x=subprocess.run([ssh,'openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True);x.check_returncode();names=json.loads(x.stdout)
import shlex
out=P/(sub or 'bmm01_evidence');out.mkdir(exist_ok=True)
archive=out/('evidence.tar' if sub=='moe01' else 'evidence.tar.gz')
with archive.open('wb') as f:
 x=subprocess.run([ssh,'openlibing-DevEnv-661495',('tar -cf - -C ' if sub=='moe01' else 'tar -czf - -C ')+shlex.quote(target)+' '+' '.join(shlex.quote(n) for n in names)],stdout=f);x.check_returncode()
with tarfile.open(archive) as t:t.extractall(out,filter='data')
print('collected',sub or 'bmm01',archive.stat().st_size)
