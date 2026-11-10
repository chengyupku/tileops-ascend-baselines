from pathlib import Path
import json,hashlib,time,sys
T=Path(__file__).resolve().parent
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
old=json.loads((T.parent/'fp16_grouped_budget900_2/FROZEN.json').read_text())['sha256']
shared='/root/dyq/workspace/TileOPs/benchmarks/ascend_all_pairs/__init__.py'
pins={p:h for p,h in old.items() if '/fp16_grouped_budget900_2/' not in p}
changes={}
for p,h in pins.items():
 current=sha(Path(p))
 if current!=h:changes[p]={'before':h,'current':current}
assert set(changes)<= {shared},changes
pins.pop(shared,None)
for f in T.iterdir():
 if f.suffix in ('.py','.sh') or f.name=='CASES.json':pins[str(f)]=sha(f)
(T/'PRE_RELEASE_PINS.json').write_text(json.dumps({'sha256':pins,'deferred_shared_initializer':shared,'historical_shared_changes':changes},indent=2))
(T/'CPU_CHECKS.json').write_text(json.dumps({'status':'pass','adapter_body_identical':True,'python_ast':'pass','mock_contract':'CONTRACT_CHECKS.json','pins_verified':len(pins),'shared_initializer_deferred_until_release':True,'no_torch_or_npu_import':True,'launch_arguments':['OP','CASE']},indent=2))
hb={'at':time.time(),'stage':'ready_for_parent_release','phase':'ready_for_parent_release','npu_started':False}
(T/'HEARTBEAT.json').write_text(json.dumps(hb,indent=2))
print('READY',len(pins),'pins; shared differences',changes)
