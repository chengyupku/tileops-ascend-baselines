from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={}
s=(r/'tools/baseline_retest32_server1_transformer_mhc_v38.py').read_text().replace('  self.binding.call(inputs,outputs,attrs)', '''  if inputs[1] is None:
   torch=self.binding.torch;x=inputs[0];n=x.shape[-2]
   identity=torch.eye(n,dtype=torch.float32,device=x.device)
   hres=identity.expand(*x.shape[:-2],n,n).contiguous()
   inputs=[inputs[0],hres,*inputs[2:]]
  self.binding.call(inputs,outputs,attrs)''')
s=s.replace("Path(adapters.__file__),","Path(adapters.__file__),Path(native.__file__),")
files['tools/baseline_retest32_server1_transformer_mhc_v39.py']=s
files['tools/baseline_retest32_server1_entry_v39.py']=(r/'tools/baseline_retest32_server1_entry_v38.py').read_text().replace('transformer_mhc_v38','transformer_mhc_v39')
files['tools/baseline_retest32_server1_python_v39.sh']=(r/'tools/baseline_retest32_server1_python_v38.sh').read_text().replace('entry_v38','entry_v39')
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v39.py').write_text(s);s=s.replace("        return (x['id']", "        return (x['id']=='320c55585d9c43be7e60' and x.get('revision')=='mhc_original_acl_v38' and '561103' in str(x.get('error'))) or (x['id']",1).replace('python_v38.sh','python_v39.sh').replace("'revision':'mhc_original_acl_v38'","'revision':'mhc_original_identity_v39'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'mhc_original_identity_v39','id':'320c55585d9c43be7e60','new_condition':'v38 plog1576125 OpDef mismatch optionalhres None; reuse original native.GraphCall existing FP32 identity expansion exactly, original ACLNN/provider/kernel unchanged'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v39')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for MHC identity v39','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v39; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
