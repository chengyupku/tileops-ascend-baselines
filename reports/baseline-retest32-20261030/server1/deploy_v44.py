from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={};pilot='0d210049c9024783c8ad'
files['tools/baseline_retest32_server1_prepermute_v44.py']='''"""Original PrePermute provider, isolated original ARM graph environment."""
import sys,types,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='benchmarks.ascend_all_pairs.ops_transformer'
package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name
sys.modules[name]=package
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=ROOT/'reports/baseline-retest32-20261030/server1/deployments/ops_prepermute/root'
import benchmarks.ascend_transformer_graph as original
from benchmarks.ascend_baselines import source_hashes
original_candidate=original._candidate
def candidate(binding,run,op,directory,adaptation):
 c=original_candidate(binding,run,op,directory,adaptation)
 def bounded_run(*args,**kwargs):
  try:return run(*args,**kwargs)
  finally:binding.cleanup()
 return dataclasses.replace(c,implementation=dataclasses.replace(c.implementation,run=bounded_run),timing_mode='host_roundtrip',source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
original._candidate=candidate
package.PROVIDERS={'MoEPrePermuteFwdOp':original.transformer_graph_prepermute}
from benchmarks.ascend_all_pairs import registry
original_factory=registry.candidate_factory
def factory(case,library):
 if library=='ops-transformer':
  assert case._call.signature.name=='MoEPrePermuteFwdOp'
  return lambda:original.transformer_graph_prepermute(case)
 return original_factory(case,library)
registry.candidate_factory=factory
'''
s=(r/'tools/baseline_retest32_server1_entry_v43.py').read_text().replace("if task['key'][:2]==['ops-transformer','MoEPostPermuteFwdOp']:","if task['key'][:2]==['ops-transformer','MoEPrePermuteFwdOp']:\n    import baseline_retest32_server1_prepermute_v44\nelif task['key'][:2]==['ops-transformer','MoEPostPermuteFwdOp']:");files['tools/baseline_retest32_server1_entry_v44.py']=s
s=(r/'tools/baseline_retest32_server1_python_v43.sh').read_text().replace('entry_v43','entry_v44').replace(' export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"',' if [ "$case_op" = MoEPrePermuteFwdOp ]; then\n  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_prepermute/vendors/prepermutepairs_transformer"\n fi\n export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"');files['tools/baseline_retest32_server1_python_v44.sh']=s
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v44.py').write_text(s);s=s.replace('key=lambda t: (',"key=lambda t: (t['id']!='"+pilot+"',",1);s=s.replace("or (t['key'][1]=='MoEPostPermuteFwdOp'","or (t['key'][1]=='MoEPrePermuteFwdOp' and (t['id']=='"+pilot+"' or latest.get('"+pilot+"',{}).get('status')=='cached')) or (t['key'][1]=='MoEPostPermuteFwdOp'",1);s=s.replace('if len(chosen)==4 or',"if len(chosen)==4 or t['id']=='"+pilot+"' or",1).replace('python_v43.sh','python_v44.sh').replace("'revision':'postpermute_original_acl_v43'","'revision':'prepermute_original_graph_v44'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'prepermute_original_graph_v44','pilot':pilot,'new_condition':'Original ARM MoeInitRoutingV2 host/source package built; original transformer_graph_prepermute unchanged; remaining6 gated on pilot'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''
import base64,json,time
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
assert (h/'deployments/ops_prepermute/vendors/prepermutepairs_transformer').is_dir()
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v44')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for PrePermute v44','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v44; draining')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
