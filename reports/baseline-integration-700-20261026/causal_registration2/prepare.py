import json,hashlib,pprint
from pathlib import Path
D=Path(__file__).resolve().parent; B=D.parent/'causal_native_build2'; A=B/'adapter'
pins={x['path']:x['sha256'] for x in json.loads((A/'PINS.json').read_text())}
local=[A/'binding.py',A/'direct_acl.py',A/'PINS.json',A/'RUNTIME_RECIPE.json',B/'ALL_CONTRACTS.json']
for p in local:pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
launch=json.loads((B/'mamba2-2.7b-decode-b1-bfloat16/NATIVE_LAUNCH.json').read_text())
loaded={x['realpath']:x['sha256'] for x in launch['loaded']}
pins.update(loaded)
params={('ops-transformer',x['op'],x['case_id']):json.dumps(x['params'],sort_keys=True,separators=(',',':'),allow_nan=False) for x in json.loads((B/'ALL_CONTRACTS.json').read_text()) if x['op']=='CausalConv1dUpdateFwdOp' and x['case_id'] in ['mamba2-2.7b-decode-b1-bfloat16','qwen3-next-decode-b64-float16']}
header='"""Exact CPU-lazy routes to unchanged, verified whole CausalConv1dUpdate APIs."""\nimport builtins, hashlib, json, os, types, tempfile\nfrom pathlib import Path\n'
for k,v in [('ROOT',str(A)),('PINS',pins),('LOCAL_FILES',[str(p) for p in local]),('LOADED_VENDOR',loaded),('CASE_PARAMS',params)]:header+=k+' = '+pprint.pformat(v,sort_dicts=True)+'\n'
header+='ROOT = Path(ROOT)\n'
(D/'integration700_causal_registration2.py').write_text(header+(D/'module_body.txt').read_text())
(D/'PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
