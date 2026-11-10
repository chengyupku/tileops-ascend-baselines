from pathlib import Path
import hashlib,json,datetime,ast
P=Path(__file__).resolve().parent
W=Path('/opt/tileops-all-pairs/workspace')
SDK=W/'.cache/all-pairs/resume22_server2_tilekernels_adaptive_norm02/sdk_view_v10'
files=[P/'adapter.py',P/'registration.py',P/'runner.py',W/'TileOPs/scripts/ascend_pair_check.py',W/'TileOPs/workloads/attention/gqa/varlen.py',W/'TileOPs/workloads/numerics.py',W/'tools/baseline_protocol.py']
files += list((P/'wheel/cann_ops_transformer/common').rglob('*'))
files += [P/'wheel/cann_ops_transformer'/x for x in ['ops/quant_flash_attn.py','ops/csrc/quant_flash_attn.cpp','op_builder/builder.py']]
files += [SDK/'lib64'/x for x in ['libopapi_transformer.so','libnnopbase.so','libopapi_math.so','libascendcl.so','libopmaster_rt.so'] if (SDK/'lib64'/x).exists()]
files += [Path('/usr/bin/g++'),Path('/usr/bin/ninja')]
rows={str(p):{'resolved':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.is_file()}
(P/'PRE_USE_IDENTITY.json').write_text(json.dumps(rows,indent=2))
s=ast.parse((P/'wheel/cann_ops_transformer/ops/quant_flash_attn.py').read_text())
(P/'API_SIGNATURES.json').write_text(json.dumps({n.name:ast.unparse(n.args) for n in s.body if isinstance(n,ast.FunctionDef) and n.name in ('quant_flash_attn','quant_flash_attn_metadata')},indent=2))
print('pinned',len(rows),'files')
