"""CPU-only stage of top-level Conv3d graph routing; does not launch device work."""
import base64,hashlib,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[3];h=Path(__file__).resolve().parent
sources=['TileOPs/benchmarks/ascend_all_pairs/ops_transformer/adapters.py','TileOPs/benchmarks/ascend_pointwise_conv.py','TileOPs/benchmarks/ascend_graph_call.py']
expected={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}
files={}
p=root/'tools/baseline_retest32_server1_transformer_graph_v25.py';s=p.read_text().replace("'Conv1dFwdOp')","'Conv1dFwdOp','Conv3dFwdOp')").replace("'Conv1dFwdOp':pointwise_conv}","'Conv1dFwdOp':pointwise_conv,'Conv3dFwdOp':pointwise_conv}");files['tools/baseline_retest32_server1_transformer_graph_v30.py']=s
s=(root/'tools/baseline_retest32_server1_entry_v27.py').read_text().replace("'Conv1dFwdOp')","'Conv1dFwdOp','Conv3dFwdOp')").replace('baseline_retest32_server1_transformer_graph_v25','baseline_retest32_server1_transformer_graph_v30');files['tools/baseline_retest32_server1_entry_v30.py']=s
s=(root/'tools/baseline_retest32_server1_python_v27.sh').read_text().replace('[ "$case_op" = Conv1dFwdOp ]; then','[ "$case_op" = Conv1dFwdOp ] || [ "$case_op" = Conv3dFwdOp ]; then').replace('baseline_retest32_server1_entry_v27.py','baseline_retest32_server1_entry_v30.py');files['tools/baseline_retest32_server1_python_v30.sh']=s
for path,s in files.items():
 if path.endswith('.py'):compile(s,path,'exec')
 (root/path).write_text(s)
encoded={p:base64.b64encode(s.encode()).decode() for p,s in files.items()}
code='expected='+repr(expected)+'\nfiles='+repr(encoded)+r'''
import base64,hashlib,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace')
actual={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in expected}
assert actual==expected,(actual,expected)
for p,s in files.items():
 data=base64.b64decode(s);dest=r/p
 if dest.exists():assert dest.read_bytes()==data
 else:
  tmp=dest.with_suffix('.stage');tmp.write_bytes(data);tmp.replace(dest)
print(json.dumps({'at':time.time(),'source_hashes':actual,'deployed':list(files),'status':'staged_only_requires_full_guarded_Conv3d_pilot','device_execution':False}))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True)
x=json.loads(r.stdout);(h/'CONV3D_STAGING.json').write_text(json.dumps(x,indent=2)+'\n');print(r.stdout)
