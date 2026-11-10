from pathlib import Path
import json,hashlib,pprint
OUT=Path(__file__).resolve().parent
R=OUT.parent/'gdn_local_native10'; P=R/'chunk_package'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
local=[R/n for n in ['CONTRACTS.json','CHUNK_CASES.json','CASE_RESOURCE_ESTIMATES.json','SOURCE_PINS.json','ADAPTER_INVARIANTS.json']]+[P/n for n in ['binding.py','direct_acl.py','resource_gate.py','entry.py','RUNTIME_RECIPE.json','RUNTIME_PINS.json','RESOURCE_ESTIMATES.json']]
runtime=json.loads((P/'RUNTIME_PINS.json').read_text())['files']
pins={str(p):h(p) for p in local};pins.update({x['path']:x['sha256'] for x in runtime});pins.update({str(R/'source'/p):v for p,v in json.loads((R/'SOURCE_PINS.json').read_text()).items()})
manifest=OUT/'FROZEN_PINS.json';manifest.write_text(json.dumps(pins,indent=2)+'\n')
contracts=json.loads((R/'CONTRACTS.json').read_text());keys=json.loads((R/'REGISTRATION_PLAN.json').read_text())['keys']
params={tuple(k):json.dumps(contracts[k[2]]['params'],sort_keys=True,separators=(',',':'),allow_nan=False) for k in keys}
recipe=json.loads((P/'RUNTIME_RECIPE.json').read_text());env=recipe['required_environment']|{'FLA_NPU_DISABLE_PTH':'1','PYTHONNOUSERSITE':'1'}
ld=['/usr/local/Ascend/cann-9.2.0-beta.2/lib64',str(R/'install/vendors/kdapairs_transformer/op_api/lib'),'/usr/local/Ascend/driver/lib64/driver','/usr/local/Ascend/driver/lib64/common','/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib','/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch_npu/lib']
libs={str(Path(x['path']).resolve()):x['sha256'] for x in runtime if '.so' in Path(x['path']).name}
header='''"""Verified registration preparation only: exact CPU-lazy GDN local routes."""\nimport builtins, hashlib, json, os, types, tempfile\nfrom pathlib import Path\nfrom types import MappingProxyType\n'''
for name,value in [('ROOT',str(P)),('MANIFEST',str(manifest)),('MANIFEST_HASH',h(manifest)),('LOCAL_FILES',tuple(map(str,local))),('CASE_PARAMS',params),('ENV',env),('LD_PATH',':'.join(ld)),('PYTHON_PATH',':'.join(recipe['python_paths'])),('LOADED_VENDOR',libs),('REQUIRED_LOADED',tuple(str(Path(recipe[x]).resolve()) for x in ['opapi','nnopbase']))]:header+=name+' = '+pprint.pformat(value,width=120)+'\n'
header+='ROOT = Path(ROOT)\n'
causal=Path('TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_causal_registration2.py').read_text()
start=causal.index('def _overlay_mapping_matches');end=causal.index('def _load(data):')
maps=causal[start:end].replace("seen != set(expected)","not set(REQUIRED_LOADED).issubset(seen)")
(OUT/'integration700_gdn_local_registration10.py').write_text(header+'\n'+maps+'\n'+(OUT/'module_body.txt').read_text())
(OUT/'append_snippet.py').write_text("\nfrom .integration700_gdn_local_registration10 import wrap as _wrap_gdn_local10\nPROVIDERS['GDNFwdOp'] = _wrap_gdn_local10(PROVIDERS.get('GDNFwdOp'))\n")
