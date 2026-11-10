from pathlib import Path
import json,shutil,hashlib
P=Path(__file__).resolve().parent;B=P/'package_local';B.mkdir(exist_ok=True)
for n in ['binding.py','direct_acl.py','entry.py']:shutil.copy2(P/'package'/n,B/n)
r=json.loads((P.parent/'gdn_local_native10/chunk_package/RUNTIME_RECIPE.json').read_text());(B/'RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2))
shutil.copy2(P.parent/'gdn_local_native10/chunk_package/RUNTIME_PINS.json',B/'RUNTIME_PINS.json')
pins=json.loads((B/'RUNTIME_PINS.json').read_text());print('pins',len(pins['files']))
(P/'LOCAL_SDK92_METADATA.json').write_text(json.dumps({'metadata':'/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/ascend_all_cann_install.info','metadata_content':Path('/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/ascend_all_cann_install.info').read_text(),'opapi':r['opapi'],'opapi_sha256':hashlib.sha256(Path(r['opapi']).read_bytes()).hexdigest(),'nnopbase':r['nnopbase'],'nnopbase_sha256':hashlib.sha256(Path(r['nnopbase']).read_bytes()).hexdigest(),'already_accepted_gdn_cases_reexecuted':False,'purpose':'reuse unchanged proven x86 native for scoped DeltaNet keys only'},indent=2))
