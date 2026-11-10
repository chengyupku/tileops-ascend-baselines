from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={}
s=(r/'tools/baseline_retest32_server1_transformer_v15.py').read_text()
s=s.replace(" return Candidate(Implementation(run),", " def bounded_run(*args,**kwargs):\n  try:return run(*args,**kwargs)\n  finally:binding.cleanup()\n return Candidate(Implementation(bounded_run),")
files['tools/baseline_retest32_server1_transformer_v37.py']=s
files['tools/baseline_retest32_server1_entry_v37.py']=(r/'tools/baseline_retest32_server1_entry_v36.py').read_text().replace('import baseline_retest32_server1_transformer_v15','import baseline_retest32_server1_transformer_v37')
files['tools/baseline_retest32_server1_python_v37.sh']=(r/'tools/baseline_retest32_server1_python_v36.sh').read_text().replace('entry_v36','entry_v37')
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v37.py').write_text(s)
s=s.replace("key=lambda t: (", "key=lambda t: (t['id']!='a3a9918f7c5e3ac9dfe7',",1)
s=s.replace("        return (x['id']", "        return (x['id']=='a3a9918f7c5e3ac9dfe7' and x.get('revision')=='add_fp16_native_v36' and 'out of memory' in str(x.get('error'))) or (x['id']",1)
s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id']=='a3a9918f7c5e3ac9dfe7' or",1).replace('python_v36.sh','python_v37.sh').replace("'revision':'add_fp16_native_v36'","'revision':'recurrent_cleanup_v37'")
files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'recurrent_cleanup_v37','id':'a3a9918f7c5e3ac9dfe7','new_condition':'Original NativeCall retains tensors/workspace across timing repetitions; complete provider invocation now ends with original synchronized cleanup inside timing; original kernel/checker unchanged; preserved OOM evidence'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v37')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for recurrent cleanup v37','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v37; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
