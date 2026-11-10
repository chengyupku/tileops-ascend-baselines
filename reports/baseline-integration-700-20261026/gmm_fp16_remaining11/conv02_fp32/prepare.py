import os,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');D=W/'reports/baseline-integration-700-20261026/transformer_dense';V=D/'gemm/private_provider';R=W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'
files=[P/'capture.py',P/'run.sh',Path('/opt/tileops-all-pairs/bin/allpairs-python'),Path('/opt/tileops-all-pairs/control/private_python.py'),Path('/opt/tileops-ci/runtime/bin/python').resolve(),Path('/usr/local/bin/npu-guard'),D/'integration700_transformer_dense_fp32.py']
files += [p for p in V.rglob('*') if p.is_file()]
files += [P/'conv1d_binding.py',W/'TileOPs/workloads/convolution.py',W/'TileOPs/src/tileops/manifest/spec/convolution.yaml',W/'TileOPs/scripts/ascend_pair_check.py',W/'TileOPs/benchmarks/api.py',W/'TileOPs/benchmarks/ascend_math_more.py',W/'TileOPs/benchmarks/ascend_baselines.py',W/'TileOPs/workloads/gemm.py',W/'TileOPs/workloads/numerics.py',W/'TileOPs/src/tileops/manifest/spec/gemm_bmm.yaml',W/'tools/baseline_protocol.py']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(P/'FROZEN.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'sha256':hashes},indent=2)+'\n')
