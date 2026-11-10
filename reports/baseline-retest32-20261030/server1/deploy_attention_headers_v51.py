from pathlib import Path
import json,time,hashlib,base64,subprocess
h=Path('reports/baseline-retest32-20261030/server1');src=Path('ascend_baselines/ops-transformer/attention/common/op_kernel');local=Path('.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer/op_impl/ai_core/tbe/classicpilot_transformer_impl/ascendc/common/op_kernel');payload={};manifest=[]
for p in src.rglob('*'):
 if p.is_file():
  rel=str(p.relative_to(src));b=p.read_bytes();assert (local/rel).read_bytes()==b;payload[rel]=base64.b64encode(b).decode();manifest.append({'path':rel,'sha256':hashlib.sha256(b).hexdigest()})
code='payload='+repr(payload)+'''
from pathlib import Path
import json,hashlib,shutil,base64,time
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1');old=h/'deployments/ops_attention/vendors/attentionpairs_transformer';new=h/'deployments/ops_attention_v51/vendors/attentionpairs_transformer';assert not new.exists();assert shutil.disk_usage(h).free>50.5*2**30+sum(p.stat().st_size for p in old.rglob('*') if p.is_file());shutil.copytree(old,new,symlinks=True)
root=new/'op_impl/ai_core/tbe/attentionpairs_transformer_impl/ascendc/common/op_kernel'
for rel,b in payload.items():
 p=root/rel;data=base64.b64decode(b)
 if p.exists():assert p.read_bytes()==data,rel
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 assert p.read_bytes()==data
fake=h/'deployments/ops_attention_v51/root/.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer';fake.parent.mkdir(parents=True);fake.symlink_to(new)
print(json.dumps({'at':time.time(),'verified_files':len(payload),'vendor':str(new)}))
'''
ssh=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495'];z=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True);(h/'ATTENTION_HEADER_CLOSURE.json').write_text(json.dumps({'deployment':json.loads(z.stdout),'files':manifest},indent=2))
files={}
for base in ['extrema','entry','python']:
 ext='.sh' if base=='python' else '.py';p=Path('tools')/f'baseline_retest32_server1_{base}_v50{ext}';s=p.read_text().replace('ops_attention/','ops_attention_v51/').replace('extrema_v50','extrema_v51').replace('entry_v50','entry_v51');q=p.with_name(p.name.replace('v50','v51'));q.write_text(s);files[str(q)]=base64.b64encode(s.encode()).decode()
p=Path('tools/run_baseline_retest32_server1.py');s=p.read_text();(h/'runner.before_v51.py').write_text(s);s=s.replace('        return (',"        return (x['key'][0]=='ops-transformer' and x['key'][1] in ('AmaxFwdOp','ReluFwdOp','ClampScalarFwdOp') and x.get('revision')=='extrema_original_graph_v50' and '500002' in str(x.get('error'))) or (",1).replace('python_v50.sh','python_v51.sh').replace("'revision':'extrema_original_graph_v50'","'revision':'attention_original_header_closure_v51'");compile(s,str(p),'exec');p.write_text(s);files[str(p)]=base64.b64encode(s.encode()).decode()
row={'at':time.time(),'revision':'attention_original_header_closure_v51','new_condition':'Original FlashAttentionScore common/op_kernel headers omitted by package; restore byte-identical local successful source closure in isolated package; retry only matching compile failure'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr(files)+'\nrow='+repr(row)+'''
from pathlib import Path
import base64,json
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 d=r/p;t=d.with_suffix('.deploy');t.write_bytes(base64.b64decode(s));t.replace(d)
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
''';subprocess.run(ssh+['python3 -'],input=code,text=True,check=True);print('deployed original headers',len(manifest))
