from pathlib import Path
import json,hashlib,datetime,os
P=Path(__file__).resolve().parent;r=json.loads((P/'ORIGINAL92_RECIPE.json').read_text());root=Path(r['vendor']);files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ['.so','.o','.json','.h','.cpp','.py']]
files.extend(Path('/usr/local/Ascend/cann-9.1.0')/s for s in ['lib64/libnnopbase.so','lib64/libopapi_math.so','lib64/libascendcl.so','lib64/libruntime.so','aarch64-linux/ascend_toolkit_install.info'])
rows=[]
for p in files:
 st=p.stat();rows.append({'path':str(p),'resolved':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'inode':st.st_ino,'device':st.st_dev})
(P/'MIGRATED_NATIVE_FREEZE.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'original_native_source_provenance':'server2 NATIVE_FREEZE.json; zero mismatches against original transformer_linear pins; unchanged ARM package copied'},indent=2))
t=P/'package/RUNTIME_PINS.json.new';t.write_text(json.dumps({'files':rows},indent=2));t.replace(P/'package/RUNTIME_PINS.json');print(len(rows))
