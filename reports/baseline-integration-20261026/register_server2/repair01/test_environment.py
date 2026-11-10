from pathlib import Path
import json,subprocess,os,sys
W=Path('/opt/tileops-all-pairs/workspace');P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer';D=W/'reports/baseline-integration-20261026/register_server2/repair01'
m=json.loads((P/'validated_manifest.json').read_text());checks=[]
for group,cfg in m['groups'].items():
 cid=next(k for k,c in m['contracts'].items() if c['group']==group);run=D/'runs/cpu-environment-test'/cid;run.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,**cfg['environment']);env['TRANSFORMER_RUN_DIR']=str(run)
 if group=='gdn':env['GDN_RUN_DIR']=str(run)
 else:env['LD_LIBRARY_PATH']=cfg['ld_library_path_prefix']
 code=f"""
import sys
sys.path.insert(0,{str(P)!r})
import validated_runtime as v
sys.path[:0]=v.MANIFEST['groups'][{group!r}]['python_paths']
v.check_environment({group!r})
assert not any(x in sys.modules for x in ['torch','torch_npu','acl'])
import os
os.environ['LD_LIBRARY_PATH']='forged-after-process-start'
try:v.check_environment({group!r})
except RuntimeError:pass
else:raise AssertionError('LD mutation accepted')
print('startup environment accepted; post-start LD mutation rejected; runtime imports zero')
"""
 r=subprocess.run(['/usr/bin/python3','-S','-B','-c',code],env=env,cwd=cfg['cwd'] if group=='gdn' else run,capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 checks.append({'group':group,'result':r.stdout.strip(),'cwd':cfg.get('cwd',str(run)),'python_paths':cfg['python_paths']})
(D/'ENVIRONMENT_CHECKS.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2));print('PASS actual CPU subprocess startup environment tests')
