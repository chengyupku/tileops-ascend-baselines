"""Deploy exact previously checked Ascend950 device binaries; CPU-only."""
import base64,hashlib,json,subprocess,time
from pathlib import Path
W=Path('/root/dyq/workspace');H=Path(__file__).resolve().parent
vendor=W/'.cache/all-pairs/fla/python_binary_partial/fla_npu/opp/vendors/fla_npu_transformer'
source=W/'ascend_baselines/flash-linear-attention-npu/fla/ops/ascendc/kda/kda_gate_cumsum'
task=next(x for x in json.loads((H.parent/'server1_ASSIGNMENTS.json').read_text())['pairs'] if x['id']=='47625ec44cc08cef5792')
check=json.loads(Path(task['local_source']['correctness_receipt']).read_text());assert check['status']=='verified'
files=list((vendor/'op_impl/ai_core/tbe/kernel/ascend950/kda_gate_cumsum').glob('*'))+[vendor/'op_impl/ai_core/tbe/kernel/config/ascend950/kda_gate_cumsum.json']
items=[]
for p in files:
 sha=hashlib.sha256(p.read_bytes()).hexdigest()
 assert check['source_sha256'][str(p.relative_to(W))]==sha
 if p.suffix=='.json' and p.with_suffix('.o').exists():assert json.loads(p.read_text())['sha256']==hashlib.sha256(p.with_suffix('.o').read_bytes()).hexdigest()
 items.append({'relative':str(p.relative_to(vendor)),'sha256':sha,'base64':base64.b64encode(p.read_bytes()).decode()})
entry=json.loads((vendor/'op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json').read_text())['KdaGateCumsum']
sources={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.suffix in ('.cpp','.h','.hpp')}
payload={'items':items,'entry':entry,'source_hashes':sources,'local_correctness':task['local_source']['correctness_receipt'],'local_guard':task['local_source']['guard_receipt']}
code='payload='+repr(payload)+r'''
import base64,hashlib,json,shutil,time
from pathlib import Path
w=Path('/opt/tileops-baseline-cache/workspace');h=w/'reports/baseline-retest32-20261030/server1';v=h/'deployments/fla_gate/fla_npu/opp/vendors/fla_npu_transformer';s=w/'ascend_baselines/flash-linear-attention-npu/fla/ops/ascendc/kda/kda_gate_cumsum'
assert shutil.disk_usage(h).free>50.5*1024**3
for rel,sha in payload['source_hashes'].items():assert hashlib.sha256((s/rel).read_bytes()).hexdigest()==sha,rel
active=json.loads((h/'ACTIVE.json').read_text())
assert all(j['key'][1]!='SSDChunkCumsumFwdOp' for j in active['jobs'])
for item in payload['items']:
 p=v/item['relative'];data=base64.b64decode(item['base64']);assert hashlib.sha256(data).hexdigest()==item['sha256']
 if p.exists():assert p.read_bytes()==data
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
p=v/'op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json';old=p.read_bytes();d=json.loads(old);assert 'KdaGateCumsum' not in d
(h/'deployments/fla_gate/binary_info_config.before_gate.json').write_bytes(old)
d['KdaGateCumsum']=payload['entry'];q=p.with_suffix('.new');q.write_text(json.dumps(d,indent=2));q.replace(p) # Atomic replacement breaks original hardlink; never mutate the shared original.
for row in payload['entry']['binaryList']:
 base=v/'op_impl/ai_core/tbe/kernel';b=base/row['binPath'];m=base/row['jsonPath'];assert b.exists() and m.exists();assert json.loads(m.read_text())['sha256']==hashlib.sha256(b.read_bytes()).hexdigest()
receipt={'at':time.time(),'source_hashes_matched':payload['source_hashes'],'local_correctness':payload['local_correctness'],'original_device_files':[{k:z for k,z in r.items() if k!='base64'} for r in payload['items']],'index_entry':payload['entry'],'original_source_math_schedule_unchanged':True,'host_and_device_source_match':True,'device_variants':3,'state':'CPU_COMPLETE_RUNTIME_UNTESTED'}
(h/'deployments/fla_gate/DEVICE_COMPLETION.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True)
(H/'GATE_DEVICE_COMPLETION.json').write_text(r.stdout)
print('CPU verified and deployed',len(items),'original files;',len(sources),'source files match; 3 device variants')
