import pathlib,json,subprocess
A=pathlib.Path(__file__).resolve().parent;S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';rows=json.loads((A/'SCOPE.json').read_text())
code=r'''
import pathlib,json,hashlib,re,datetime
W=pathlib.Path('/opt/tileops-all-pairs/workspace');D=W/'reports/baseline-integration-700-20261026/transformer_attention_misc';rows=ROWS
out=[];seen=set()
for r in rows:
 for p in list((D/'runs'/r['case_id']).rglob('*.log')) + list((D/'ASCEND_PROCESS_LOG_PATH').rglob('*.log')):
  if p in seen:continue
  seen.add(p);hits=[];h=hashlib.sha256()
  with p.open('rb') as f:
   for i,b in enumerate(f):
    h.update(b);s=b.decode(errors='replace').rstrip()
    if (('flashattentionscore' in s.lower() or 'flash_attention_score' in s.lower()) and any(t in s.lower() for t in ['launch','execute','classicpilot_attention_transformer','kernelname','kernel_name','compile command'])):
     hits.append({'line':i+1,'text':s})
  if hits:out.append({'path':str(p),'sha256':h.hexdigest(),'matches':hits})
print(json.dumps(out))
'''.replace('ROWS',repr(rows))
p=subprocess.run([S,'openlibing-DevEnv-055409','PYTHONDONTWRITEBYTECODE=1 python3 -S -'],input=code,text=True,capture_output=True,timeout=160);(A/'NATIVE_LAUNCH.json').write_text(p.stdout);print(p.returncode,p.stderr,'bytes',len(p.stdout))
