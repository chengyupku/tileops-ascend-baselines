import json,hashlib,pathlib
P=pathlib.Path(__file__).resolve().parent; D=P.parent/'attention_direct_api5'
read=lambda p:json.loads(p.read_text())
pins=read(D/'REGISTRATION_PINS.json')
src='/opt/tileops-all-pairs/workspace/.cache/all-pairs/root_server2_20261022/source_templates/ascend_baselines/ops-transformer/'
for k,v in read(P.parent/'attention_generated_recovery5/RELEVANT_EXPECTED_SOURCE_PINS.json').items():
 assert src+k not in pins or pins[src+k]==v
 pins[src+k]=v
params={};loaded={};evidence={}
for row in read(P/'SCOPE.json'):
 key=(row['baseline'],row['operator'],row['case_id']); cid=key[2]
 c=read(D/(cid+'.checker.json')); params[key]=json.dumps(c['case_snapshot']['parameters'],sort_keys=True,separators=(',',':'),allow_nan=False)
 r=read(D/(cid+'.receipt.json'));loaded.update(r['loaded_vendor_libraries'])
 for suffix in ['checker.json','receipt.json','native.json','maps','guard.json']:
  p=D/(cid+'.'+suffix); evidence[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 n=read(D/(cid+'.native.json'))
 for k,v in n['loaded_files_sha256'].items():
  if k.endswith('/libnnopbase.so'):pins[k]=v
module='''"""Five exact, lazy original FlashAttn routes; CPU-safe construction."""
import builtins, hashlib, json, os, types
from pathlib import Path
from types import MappingProxyType
'''
module+='ROOT = Path('+repr(str(D))+')\n'
module+='CASE_PARAMS = MappingProxyType('+repr(params)+')\n'
module+='PINS = MappingProxyType('+repr(pins)+')\n'
module+='LOADED_VENDOR = MappingProxyType('+repr(loaded)+')\n'
module+='EVIDENCE_PINS = MappingProxyType('+repr(evidence)+')\n'
module+='SOURCE = '+repr(src+'attention/flash_attn/torch_extension/flash_attn.py')+'\n'
module+=(P/'module_body.txt').read_text()
(P/'integration700_attention_registration5.py').write_text(module)
(P/'PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
print('pins',len(pins),'source',sum(k.startswith(src) for k in pins),'loadedvendor',len(loaded))
