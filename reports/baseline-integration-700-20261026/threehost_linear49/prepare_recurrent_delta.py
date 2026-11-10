from pathlib import Path
import json,shutil,subprocess,hashlib
P=Path(__file__).resolve().parent;B=P/'package_recurrent';B.mkdir(exist_ok=True)
for n in ['direct_acl.py','entry.py']:shutil.copy2(P/'package'/n,B/n)
r=json.loads((P.parent/'transformer_linear/recurrent_package/RUNTIME_RECIPE.json').read_text());v=Path(r['vendor']);ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';D='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_linear49'
a=subprocess.run([ssh,'openlibing-DevEnv-055409',f'tar -czhf - -C {v.parent} {v.name}'],capture_output=True,check=True);(P/'recurrent_custom92_aarch64.tar.gz').write_bytes(a.stdout)
subprocess.run([ssh,'openlibing-DevEnv-661495',f'mkdir -p {D}/native_recurrent92 && tar -xzf - -C {D}/native_recurrent92'],input=a.stdout,check=True)
r={'opapi':D+'/native_recurrent92/'+v.name+'/op_api/lib/libcust_opapi.so','nnopbase':'/usr/local/Ascend/cann-9.1.0/lib64/libnnopbase.so','vendor':D+'/native_recurrent92/'+v.name}
(B/'RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2));(B/'RUNTIME_PINS.json').write_text('{"files": []}')
(P/'RECURRENT_MIGRATION.json').write_text(json.dumps({'original_vendor':str(v),'private_vendor':r['vendor'],'archive_sha256':hashlib.sha256(a.stdout).hexdigest(),'source_binding':'../transformer_linear/recurrent_package/binding.py','mathematical_mapping':'DeltaNet ungated case via original complete RecurrentKda at constant zero log gate, preserve all original tensor values and FP32 state'},indent=2))
print('original recurrent package migrated',flush=True)
