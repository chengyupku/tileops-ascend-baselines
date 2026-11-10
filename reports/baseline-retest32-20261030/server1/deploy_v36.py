from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={}
s=(r/'tools/baseline_retest32_server1_transformer_add_v35.py').read_text();s=s.replace("    assert case._call.signature.name=='AddFwdOp'", "    if case._call.signature.name=='AddFwdOp' and case.params['input'][1]=='float16':\n        from benchmarks.ascend_transformer_add import transformer_add\n        return lambda:transformer_add(case)\n    assert case._call.signature.name=='AddFwdOp'",1);files['tools/baseline_retest32_server1_transformer_add_v36.py']=s
files['tools/baseline_retest32_server1_entry_v36.py']=(r/'tools/baseline_retest32_server1_entry_v35.py').read_text().replace('transformer_add_v35','transformer_add_v36')
files['tools/baseline_retest32_server1_python_v36.sh']=(r/'tools/baseline_retest32_server1_python_v35.sh').read_text().replace('entry_v35','entry_v36')
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v36.py').write_text(s);s=s.replace("t['id'] not in ('1897", "t['id']!='6b14ecb7247c290836c8',t['id'] not in ('1897",1);s=s.replace("or (t['key'][1]=='AddFwdOp'", "or (t['key'][1]=='AddFwdOp' and t['case']['params']['input'][1]=='float16' and (t['id']=='6b14ecb7247c290836c8' or latest.get('6b14ecb7247c290836c8',{}).get('status')=='cached')) or (t['key'][1]=='AddFwdOp'",1);s=s.replace('if len(chosen)==4 or','if len(chosen)==4 or t["id"]=="6b14ecb7247c290836c8" or',1).replace('python_v35.sh','python_v36.sh').replace("'revision':'add_routing_acl_v35'","'revision':'add_fp16_native_v36'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'add_fp16_native_v36','pilot':'6b14ecb7247c290836c8','new_condition':'Original LayerNormAddAddQuant sourceclosure CPUcompiled+linked forARM torch210 with pairedCANN9.1SDK,originalauto-sync kept,receiptSHAverified;direct originaltransformer_add provider,remaining3releasedonlyafterpilotcached'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''\nfrom pathlib import Path
import base64,json,time,hashlib
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1';cache=r/'.cache/ascend-classic/native-pilot-20261020/ops-transformer-add';receipt=json.loads((cache/'build_receipt.json').read_text());assert hashlib.sha256((cache/'libclassic_transformer_add_original.so').read_bytes()).hexdigest()==receipt['binary_sha256']
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v36')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for FP16 Add original native pilot v36','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v36')
''';subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
