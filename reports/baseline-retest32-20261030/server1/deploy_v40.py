from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={}
s=(r/'tools/baseline_retest32_server1_transformer_mhc_v39.py').read_text()
s=s.replace('from benchmarks.ascend_all_pairs.ops_transformer.aclnn_bridge import HostCall',"import benchmarks.ascend_graph_call as graph\ngraph.WORKSPACE=ROOT/'reports/baseline-retest32-20261030/server1/deployments/ops_graph/root'")
s=s.replace("self.binding=HostCall('MhcPost',self.vendor)","self.binding=graph.GraphCall(library)")
s=s.replace('self.binding.call(inputs,outputs,attrs)',"self.binding.call(op,inputs,outputs,attrs)")
s=s.replace('Path(native.__file__),','Path(native.__file__),Path(graph.__file__),')
files['tools/baseline_retest32_server1_transformer_mhc_v40.py']=s
files['tools/baseline_retest32_server1_entry_v40.py']=(r/'tools/baseline_retest32_server1_entry_v39.py').read_text().replace('transformer_mhc_v39','transformer_mhc_v40')
files['tools/baseline_retest32_server1_python_v40.sh']=(r/'tools/baseline_retest32_server1_python_v39.sh').read_text().replace('entry_v39','entry_v40')
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v40.py').write_text(s);s=s.replace("        return (x['id']", "        return (x['id']=='320c55585d9c43be7e60' and x.get('revision')=='mhc_original_identity_v39' and '561000' in str(x.get('error'))) or (x['id']",1).replace('python_v39.sh','python_v40.sh').replace("'revision':'mhc_original_identity_v39'","'revision':'mhc_original_graph_v40'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'mhc_original_graph_v40','id':'320c55585d9c43be7e60','new_condition':'v39 native binary missing tilingkey5 entry; original graph compiler JIT route with unchanged source/vendor and original identity mapping; no flags/kernel edits'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v40')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for MHC graph v40','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v40; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
