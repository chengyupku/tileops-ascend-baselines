from pathlib import Path
import json,shutil,hashlib
P=Path(__file__).resolve().parent;W=P.parents[2];C=W/'.cache/all-pairs/remote_tla_compiler';S=W/'.cache/all-pairs/remote_native_deepgemm_deepep'
files=[C/'handoff/build_receipt.json',*[C/'metadata'/n for n in ['compile_input_hashes.json','source_final_verification.json','official_tree_comparison.json','compiled_submodule_comparisons.json']],S/'sdk-920b2/assembly_receipt.json',S/'sdk-920b2/assembly_extend_receipt.json']
records=[]
for i,f in enumerate(files):
 if not f.exists():continue
 d=P/'provenance'/f'{i}_{f.name}';d.parent.mkdir(exist_ok=True);shutil.copyfile(f,d);records.append(dict(original=str(f),saved=str(d),sha256=hashlib.sha256(d.read_bytes()).hexdigest()))
for f in (S/'attempts').glob('*.json'):
 if any(k in f.name for k in ['download','sdk','extract','920']):
  d=P/'provenance'/f.name;shutil.copyfile(f,d);records.append(dict(original=str(f),saved=str(d),sha256=hashlib.sha256(d.read_bytes()).hexdigest()))
(P/'PROVENANCE_INDEX.json').write_text(json.dumps(records,indent=2));print(len(records))
