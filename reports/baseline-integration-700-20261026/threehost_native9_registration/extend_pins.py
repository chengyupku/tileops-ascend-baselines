import json,subprocess,shlex,hashlib
from pathlib import Path
O=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration');S=O.parent/'threehost_native_completion9'
paths=set()
for cid in ['llama-8b-prefill-bfloat16','square-1k-nk-float8_e4m3fn-bfloat16']:
 d=json.loads((S/'server1'/cid/'result/EXECUTION.json').read_text());paths.update(p for p in d['loaded_files'] if '.so' in p and p.startswith('/'))
script='''import json,hashlib\nfrom pathlib import Path\npaths=set(%r)\nsite=Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages')\npaths.update(str(p) for p in site.glob('*.pth'))\npaths.update(str(site/p/'__init__.py') for p in ['torch','torch_npu'])\nvendor=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider')\npaths.update(str(p) for p in vendor.rglob('*.so'))\nout={}\nfor p in sorted(paths):\n with open(p,'rb') as f:\n  h=hashlib.sha256()\n  for b in iter(lambda:f.read(1048576),b''):h.update(b)\n out[p]=h.hexdigest()\nprint(json.dumps(out))\n'''%sorted(paths)
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
new=json.loads(subprocess.check_output([ssh,'openlibing-DevEnv-661495','python3 -S -c '+shlex.quote(script)],text=True))
d=json.loads((O/'PINS.json').read_text());d['runtime'].update(new);d['provenance']+='; actual successful FP8/BF16 EXECUTION mapped DSO set and runtime entrypoints pinned read-only at registration (not retroactive receipt hashes)'
(O/'PINS.json').write_text(json.dumps(d,indent=2)+'\n')
p=O/'registration.py';src=p.read_text();lines=src.splitlines();lines=[('DATA_HASHES = '+repr({n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['CONTRACTS.json','PINS.json','FROZEN_fp8.json','FROZEN_bf16arm2.json']})) if line.startswith('DATA_HASHES =') else line for line in lines];p.write_text('\n'.join(lines)+'\n');print('runtime pins',len(d['runtime']))
