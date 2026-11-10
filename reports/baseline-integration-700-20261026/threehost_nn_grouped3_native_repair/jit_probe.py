from pathlib import Path
import json,hashlib,subprocess
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11')
for f in [P/'vendor/allpairs_nn/op_impl/ai_core/tbe/allpairs_nn_impl/dynamic/mat_mul_v3.py']:
 print('JIT',f,hashlib.sha256(f.read_bytes()).hexdigest());print(f.read_text())
A=P/'remote_bf16_aclnn02'
for base in [A/'ASCEND_CACHE_PATH',A/'ASCEND_WORK_PATH', A/'tmp']:
 fs=[f for f in base.rglob('*') if f.is_file()];print('CACHE_FILES',base,len(fs))
 for f in fs:
  if f.suffix in ['.json','.o','.so']:
   print('FILE',f,f.stat().st_size,hashlib.sha256(f.read_bytes()).hexdigest())
   if f.suffix=='.json':print(f.read_text()[:18000])
for f in A.glob('ASCEND_PROCESS_LOG_PATH/debug/plog/*'):
 for l in f.read_text(errors='replace').splitlines():
  if any(x in l for x in ['24578','BinaryGetFunction','MatMulV3_', 'mat_mul_v3.o']): print('LOG',l[:1600])
