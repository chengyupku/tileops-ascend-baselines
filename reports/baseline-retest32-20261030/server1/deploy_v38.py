from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={}
s=(r/'tools/baseline_retest32_server1_transformer_v37.py').read_text()
s=s.replace('from native_call import NativeCall','''import os
from benchmarks.ascend_all_pairs.ops_transformer.aclnn_bridge import HostCall
class NativeCall:
 def __init__(self,library,vendor_name=None):
  assert library=='ops-transformer'
  self.vendor=Path(os.environ['OLD1076_TRANSFORMER_VENDOR'])
  self.binding=HostCall('MhcPost',self.vendor)
 def call(self,op,inputs,outputs,attrs):
  assert op=='MhcPost' and attrs=={}
  self.binding.call(inputs,outputs,attrs)
 def cleanup(self):self.binding.cleanup()''')
s=s.replace("ROOT/'tools/old1076_transformer/native_call.py'","ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/aclnn_bridge.py',ROOT/'TileOPs/benchmarks/ascend_math_more.py'")
s=s.replace("package.PROVIDERS={op:adapters.recurrent_family for op in ('DeltaNetRecurrentFwdOp','GDNFwdOp')}","package.PROVIDERS={'MHCPostFwdOp':adapters.mhc_post}").replace('lambda: adapters.recurrent_family(case)','lambda: adapters.mhc_post(case)')
files['tools/baseline_retest32_server1_transformer_mhc_v38.py']=s
s=(r/'tools/baseline_retest32_server1_entry_v37.py').read_text();s=s.replace("if task['key'][:2]==['ops-transformer','AddFwdOp']:","if task['key'][:2]==['ops-transformer','MHCPostFwdOp']:\n    import baseline_retest32_server1_transformer_mhc_v38\nelif task['key'][:2]==['ops-transformer','AddFwdOp']:");files['tools/baseline_retest32_server1_entry_v38.py']=s
files['tools/baseline_retest32_server1_python_v38.sh']=(r/'tools/baseline_retest32_server1_python_v37.sh').read_text().replace('entry_v37','entry_v38').replace('[ "$case_op" = AddFwdOp ];','[ "$case_op" = AddFwdOp ] || [ "$case_op" = MHCPostFwdOp ];')
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v38.py').write_text(s);s=s.replace('key=lambda t: (',"key=lambda t: (t['id']!='320c55585d9c43be7e60',",1);s=s.replace("or (t['key'][1]=='AddFwdOp'","or (t['key'][1]=='MHCPostFwdOp' and (t['id']=='320c55585d9c43be7e60' or latest.get('320c55585d9c43be7e60',{}).get('status')=='cached')) or (t['key'][1]=='AddFwdOp'",1);s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id']=='320c55585d9c43be7e60' or",1).replace('python_v37.sh','python_v38.sh').replace("'revision':'recurrent_cleanup_v37'","'revision':'mhc_original_acl_v38'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'mhc_original_acl_v38','pilot':'320c55585d9c43be7e60','new_condition':'MhcPost original source46files deployed/SHAverified; original native public5tensor ABI present; reuse original HostCall and adapters.mhc_post, per-call cleanup timed; remaining2 gated on full pilot acceptance'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v38')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for MHC original ACL v38','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v38; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
