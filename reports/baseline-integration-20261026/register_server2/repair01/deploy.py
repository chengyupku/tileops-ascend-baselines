import subprocess,tarfile,io
from pathlib import Path
D=Path(__file__).resolve().parent
ssh=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409']
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as t:
 for n in ['validated_runtime.py','attention_adapter.py','gdn_adapter.py','launch_verified.py']:t.add(D/n,arcname=n)
subprocess.run(ssh+['tar -xf - -C /opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer'],input=buf.getvalue(),check=True)
code="""
from pathlib import Path
import json
P=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer');D=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-20261026/register_server2/repair01')
m=json.loads((D/'manifest_draft.json').read_text());m['run_root']=str(D/'runs')
(P/'validated_manifest.json').write_text(json.dumps(m,indent=2)+'\\n')
import sys
sys.dont_write_bytecode=True;sys.path.insert(0,str(P))
import validated_runtime as v
for group in ['attention','gdn']:
 try:v.verify_pins(group);print(group,'pins and directories PASS')
 except Exception as e:print(group,type(e).__name__,str(e))
"""
subprocess.run(ssh+['python3 -S -B -'],input=code.encode(),check=True)
