from pathlib import Path
import json,hashlib,subprocess
W=Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/threehost_nn11';B=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build'
for base in [P/'vendor',B/'install3',B/'cache',B/'work']:
 if base.exists():
  fs=[f for f in base.rglob('*') if f.is_file() and ('matmul' in str(f).lower() or 'mat_mul' in str(f).lower())];print('FILES',str(base),len(fs)); print('\n'.join(str(f) for f in fs[:100]))
for n in ['ORIGINAL_JIT_REUSE.json','NATIVE_READY.json','BUILD_COMMANDS.md','configure.sh','INITIALIZATION_BOUNDARY.md']:
 f=B/n
 if f.exists():print('EVIDENCE',n,f.read_text()[:16000])
S=B/'source'
for rel in ['matmul/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_key.h','matmul/mat_mul_v3/op_kernel/arch35/mat_mul_v3_tiling_key_public.h','matmul/mat_mul_v3/op_kernel/arch35/mat_mul_v3.cpp','matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_key.cpp','matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp','matmul/mat_mul_v3/CMakeLists.txt']:
 f=S/rel
 if f.exists(): print('SOURCE',rel,hashlib.sha256(f.read_bytes()).hexdigest(),'\n',f.read_text()[:26000])
