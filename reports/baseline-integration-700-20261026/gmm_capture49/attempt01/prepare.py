import os,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');D=W/'reports/baseline-integration-700-20261026/transformer_dense';V=D/'gemm/private_provider';R=W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'
files=[P/'capture.py',P/'run.sh',Path('/opt/tileops-all-pairs/bin/allpairs-python'),Path('/opt/tileops-all-pairs/control/private_python.py'),Path('/opt/tileops-ci/runtime/bin/python').resolve(),Path('/usr/local/bin/npu-guard'),D/'integration700_transformer_dense_gmm.py']
files += [p for p in V.rglob('*') if p.is_file()]
for root in [W/'TileOPs/workloads',W/'TileOPs/benchmarks']:
 files += list(root.rglob('*.py'))
files += [W/'TileOPs/scripts/ascend_pair_check.py',W/'tools/baseline_protocol.py']
for root in [R/'torch',R/'torch_npu']:
 files += [p for p in root.rglob('*') if p.is_file() and (p.suffix in ('.py','.so') or '.so.' in p.name)]
files += [p for p in Path('/usr/local/Ascend/cann-9.1.0/lib64').glob('*.so*') if p.is_file()]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(P/'FROZEN.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes},indent=2)+'\n')
