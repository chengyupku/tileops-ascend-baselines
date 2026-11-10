from pathlib import Path
import json,hashlib,platform,os,ast
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_attention_mla_mha31');W=P.parents[2];V=W/'ascend_baselines/tilelang-ascend'
files=[V/'examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py',V/'examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py']+list((W/'.cache/all-pairs/remote_tla_compiler/handoff/lib').glob('*.so'))
r={'machine':platform.machine(),'assets':[],'frozen_mismatches':[]}
for p in files:
 b=p.read_bytes();elf=int.from_bytes(b[18:20],'little') if b[:4]==b'\x7fELF' else None
 r['assets'].append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'elf_machine':elf});assert elf in [None,183]
for n,h in json.loads((P/'FROZEN_HASHES.json').read_text()).items():
 if hashlib.sha256((P/n).read_bytes()).hexdigest()!=h:r['frozen_mismatches'].append(n)
for p in P.glob('*.py'):ast.parse(p.read_text())
assert not r['frozen_mismatches'];assert len(r['assets'])>=4
(P/'PREFLIGHT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
