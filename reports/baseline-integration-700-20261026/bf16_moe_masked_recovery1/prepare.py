from pathlib import Path
import ast,json,hashlib,time,subprocess
T=Path(__file__).resolve().parent;D=T.parent/'bf16_local_gemm18';W=Path('/root/dyq/workspace')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(T.parent/'bf16_moe_hadamard_parent10/adapter.py')=='4d843e71be9049844242420664c73df3907912edc4561ce6e42ff294bc4d0bb0'
assert sha(T/'adapter.py')==sha(T.parent/'bf16_moe_hadamard_native10/adapter.py')
for p in T.glob('*.py'):ast.parse(p.read_text())
launch=(T/'launch.sh').read_text();capture=(T/'capture.py').read_text()
assert '/root/dyq/workspace/.venv/bin/python -B "$P/capture.py" "$P/result" MoEGroupedGemmFwdOp ds-v3-ep8-masked-bfloat16' in launch
assert 'OP=sys.argv[2];CASE_ID=sys.argv[3]' in capture
assert "'--library','ops-transformer'" in capture and 'case._call.indices' not in capture
subprocess.run(['bash','-n',str(T/'launch.sh'),str(T/'environment.sh')],check=True)
old=json.loads((D/'square-1k-bfloat16/FROZEN.json').read_text())['sha256']
pins={p:h for p,h in old.items() if '/install/' in p or '/site-packages/' in p or '/TileOPs/' in p or p.endswith('/tools/baseline_protocol.py') or p.endswith('/binding.py') or p=='/usr/local/bin/npu-guard'}
pins.update(json.loads((D/'SOURCE_PINS.json').read_text()))
changed={p:dict(old=h,current=sha(p)) for p,h in pins.items() if sha(p)!=h}
# Shared registry changed by parent publication; freeze current only after release.
assert (T/'PARENT_RELEASE.json').exists()
allowed=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py'
assert all(p==str(allowed) for p in changed),changed
for p in changed:pins[p]=sha(p)
for p in list(T.glob('*.py'))+[T/'CASES.json',T/'launch.sh',T/'environment.sh',T/'PARENT_RELEASE.json',T.parent/'bf16_moe_hadamard_parent10/adapter.py',T.parent/'bf16_moe_hadamard_parent10/cpu_checks.py',T.parent/'bf16_moe_hadamard_native10/adapter.py']:
 pins[str(p)]=sha(p)
# Pin deployed metadata, original sources, SDK environment and runtime ELF bytes.
sdk=Path('/usr/local/Ascend/cann-9.2.0-beta.2')
for p in [sdk/'set_env.sh']+list((sdk/'aarch64-linux/lib64').glob('*.so*'))+list((sdk/'x86_64-linux/lib64').glob('*.so*')):
 if p.is_file():pins[str(p)]=sha(p)
for name in ['DEPLOYMENT_MANIFEST.json','SOURCE_PINS.json','ORIGINAL_BOUNDARY.json','DEVICE_COMMAND.json']:pins[str(D/name)]=sha(D/name)
release=json.loads((T.parent/'DEPLOYMENT_RELEASE_BF16_V5_20261026T2333.json').read_text())
for k in ['binding','native_host','device_object','original_generated_recipe']:
 v=release[k];assert sha(v['path'])==v['sha256'];pins[v['path']]=v['sha256']
(T/'FROZEN.json').write_text(json.dumps(dict(sha256=pins,release=release,shared_registry_updates=changed),indent=2))
(T/'PREPARATION.json').write_text(json.dumps(dict(pins=len(pins),launch_arguments_checked=True,adapter_exact=True,shared_registry_updates=changed),indent=2))
(T/'HEARTBEAT.json').write_text(json.dumps(dict(stage='ready_for_release',time=time.time())))
print('pins',len(pins))
