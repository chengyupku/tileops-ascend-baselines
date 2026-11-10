import json,hashlib,datetime
from pathlib import Path
O=Path(__file__).resolve().parent; P=O.parent/'gdn_capacity_chunk16'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
scope=json.loads((O/'SCOPE.json').read_text());ids=[s['case_id'] for s in scope];assert len(set(ids))==16
contracts=json.loads((P/'CONTRACTS.json').read_text());params={c:json.dumps(contracts[c]['params'],sort_keys=True,separators=(',',':'),allow_nan=False) for c in ids}
frozen=json.loads((P/'FROZEN_ADAPTER.json').read_text())['files'];pins={str(Path(p).relative_to(P)):v for p,v in frozen.items()}
for c in ids:pins[c+'/child.sh']=sha(P/c/'child.sh')
for n,v in pins.items():assert sha(P/n)==v
recipe=json.loads((P/'chunk_package/RUNTIME_RECIPE.json').read_text())
# Filter metadata in memory only; never hash the unrelated runtime catalog.
catalog=json.loads((P/'chunk_package/RUNTIME_PINS.json').read_text())['files']
runtime={x['path']:x['sha256'] for x in catalog if ('/source_templates/' in x['path'] and '/chunk_gated_delta_rule/op_' in x['path'] and '/include/' not in x['path']) or x['path'] in (recipe['opapi'],recipe['nnopbase'])}
loaded={}
for c in ids:
 for x in json.loads((P/c/'loaded-libraries.json').read_text()):
  assert x['path'] not in loaded or loaded[x['path']]==x['sha256']
  loaded[x['path']]=x['sha256']
runtime.update(loaded)
for p,x in json.loads((P/'DEPLOYMENT_VERIFIED.json').read_text()).items():
 if p.startswith(recipe['vendor']+'/'):assert x['actual']==x['expected'];runtime[p]=x['expected']
base=(O.parent/'gdn_recurrent_registration1/integration700_gdn_recurrent_registration1.py').read_text();body=base[base.index('def _checked'):]
body=body.replace("case.id != CASE_ID or json.dumps(case.params,sort_keys=True,separators=(',',':'),allow_nan=False) != PARAMS_JSON", "case.id not in CASE_PARAMS or json.dumps(case.params,sort_keys=True,separators=(',',':'),allow_nan=False) != CASE_PARAMS[case.id]")
body=body.replace("'recurrent_package'","'chunk_package'").replace('id=CASE_ID, params=json.loads(PARAMS_JSON)','id=case.id, params=json.loads(CASE_PARAMS[case.id])').replace('case.id == CASE_ID','case.id in CASE_PARAMS')
header='"""Frozen exact16 original whole ChunkGatedDeltaRule registration; no test switches."""\nimport builtins, hashlib, json, os, types\nfrom pathlib import Path\nfrom types import MappingProxyType\n'
header+=f'ROOT = Path({str(P)!r})\nCASE_PARAMS = MappingProxyType({params!r})\nFILE_PINS = MappingProxyType({pins!r})\nRUNTIME_PINS = MappingProxyType({runtime!r})\nENV = MappingProxyType({recipe["required_environment"]!r})\nLD_PREFIX = {tuple(recipe["ld_library_path_order"])!r}\n'
(O/'integration700_gdn_chunk_registration16.py').write_text(header+body)
write('PIN_SELECTION.json',{'runtime':runtime,'file_pins':pins,'loaded_library_count':len(loaded),'runtime_count':len(runtime),'selection':'Chunk source/deployment/metadata candidates plus union of original 16 loaded libraries; no unrelated catalog file hashing','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
write('NATIVE_EVIDENCE.json',{'api':'aclnnChunkGatedDeltaRule','opapi':recipe['opapi'],'opapi_sha256':runtime[recipe['opapi']],'nnopbase':recipe['nnopbase'],'source_line_evidence':str(P/'SOURCE_LINE_EVIDENCE.json'),'original_delivery_audit':str(P/'DELIVERY_AUDIT.json'),'cases':{c:{'loaded_library_evidence':str(P/c/'loaded-libraries.json'),'opapi_loaded':recipe['opapi'] in {x['path'] for x in json.loads((P/c/'loaded-libraries.json').read_text())}} for c in ids},'boundary':'Full original ACLNN API and actual loaded opapi are established. Source lines are static, and pinned kernel metadata/binaries are candidates, not observed runtime binary selection. No binary-specific launch log is present in original delivered evidence. No success case rerun or whole unknown backend success claim.','kernel_or_compiler_changes':False})
print('pins',len(pins),len(runtime))
