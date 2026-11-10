from pathlib import Path
import json,hashlib,re,shutil
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace')
frozen=json.loads((P/'FROZEN.json').read_text())['sha256'];logs=list((P/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'))
lines=[l for p in logs for l in p.read_text(errors='replace').splitlines() if any(s in l for s in ['aclnnGroupedMatmulV5','Register Kernel succ','kernelName','LaunchKernel','launch kernel'])]
(P/'native_dispatch_excerpt.log').write_text('\n'.join(lines)+'\n')
selected=set(re.findall(r'bin path: (\S+GroupedMatmul_\w+\.o)','\n'.join(lines)))
assert len(selected)==1,selected
files=[W/'TileOPs/scripts/ascend_pair_check.py',W/'TileOPs/workloads/gemm.py',W/'TileOPs/workloads/numerics.py',W/'TileOPs/benchmarks/ascend_math_more.py',W/'TileOPs/benchmarks/ascend_baselines.py',W/'TileOPs/benchmarks/api.py',W/'tools/baseline_protocol.py',W/'reports/baseline-integration-700-20261026/transformer_dense/integration700_transformer_dense_gmm.py']+[Path(p) for p in selected]
S=P/'source_snapshot';S.mkdir()
for p in files:
 assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen[str(p)]
 shutil.copyfile(p,S/p.name)
execution=json.loads((P/'EXECUTION.json').read_text())
loaded={p:frozen[p] for p in set(execution['loaded_files']) if p in frozen}
(P/'NATIVE_DISPATCH.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'selected_binary':{p:frozen[p] for p in selected},'loaded_frozen_files':loaded,'frozen_unchanged':execution['frozen_unchanged'],'full_logs':[str(p.relative_to(P)) for p in logs]},indent=2)+'\n')
(P/'REMOTE_ARTIFACTS.json').write_text(json.dumps({str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and p.name!='REMOTE_ARTIFACTS.json'},indent=2)+'\n')
