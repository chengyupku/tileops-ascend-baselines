from pathlib import Path
import ast,json,hashlib,subprocess,time
T=Path(__file__).resolve().parent;R=T.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((R/'bf16_moe_masked_recovery1/FROZEN.json').read_text())['sha256']
pins={p:sha(p) for p in old if '/bf16_moe_masked_recovery1/' not in p}
changes={p:{'old':old[p],'current':h} for p,h in pins.items() if old[p]!=h}
assert all(p.endswith('/ops_transformer/__init__.py') for p in changes),changes
for p in T.glob('*.py'):ast.parse(p.read_text())
subprocess.run(['bash','-n',str(T/'launch.sh'),str(T/'environment.sh')],check=True)
assert '"$P/$CASE/result" MoEGroupedGemmFwdOp "$CASE"' in (T/'launch.sh').read_text()
assert 'case._call.indices.items()' in (T/'capture.py').read_text()
assert sha(T/'adapter.py')==sha(R/'bf16_moe_masked_recovery1/adapter.py')
for p in list(T.glob('*.py'))+[T/n for n in ['CASES.json','PARENT_RELEASE.json','launch.sh','environment.sh']]:pins[str(p)]=sha(p)
frozen=dict(sha256=pins,current_initializer_changes=changes,pin_list_source=str(R/'bf16_moe_masked_recovery1/FROZEN.json'))
for row in json.loads((T/'CASES.json').read_text()):
 p=T/row['key'][2];(p/'result').mkdir(parents=True,exist_ok=True);(p/'FROZEN.json').write_text(json.dumps(frozen,indent=2))
(T/'FROZEN.json').write_text(json.dumps(frozen,indent=2))
(T/'CPU_CHECKS.json').write_text(json.dumps(dict(ast='pass',bash_syntax='pass',launch_positional_args='pass',manifest_comparison='actual_call_indices',adapter='byte identical successful masked recovery',pins=len(pins),changes=changes),indent=2))
(T/'HEARTBEAT.json').write_text(json.dumps(dict(time=time.time(),stage='prepared')))
print(len(pins),changes)
