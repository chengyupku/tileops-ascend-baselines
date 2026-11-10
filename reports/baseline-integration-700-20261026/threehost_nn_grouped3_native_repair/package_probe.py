from pathlib import Path
import json,hashlib,subprocess
B=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build');P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11')
def info(f):
 s=f.stat();return dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),size=s.st_size,inode=s.st_ino,device=s.st_dev)
f=Path('/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/mat_mul_v3/MatMulV3_ND_ND_ND_ND_FP32_FP32_FP32_FP32_high_performance.json')
if f.exists():
 print('SYSTEM_META',info(f));d=json.loads(f.read_text());print('METADATA',json.dumps(d)[:35000])
 for o in f.parent.glob(f.stem+'*'):
  if o.suffix!='.json':
   print('BIN',info(o));r=subprocess.run(['readelf','-sW',str(o)],capture_output=True,text=True); lines=[l for l in r.stdout.splitlines() if '24578' in l];print('SYMBOL24578',lines,'rc',r.returncode)
for base in [P/'vendor/allpairs_nn/op_impl/ai_core/tbe/kernel',B/'install3/vendors/allpairs_nn/op_impl/ai_core/tbe/kernel']:
 print('CUSTOM_KERNEL_DIR',str(base),base.exists(),sum(1 for f in base.rglob('*') if f.is_file()) if base.exists() else 0)
if B.exists():
 for rel in ['source/build.py','source/cmake/intf.cmake','source/CMakeLists.txt','build/Makefile']:
  f=B/rel
  if f.exists():
   print('BUILD_TARGET',rel)
   for i,l in enumerate(f.read_text().splitlines(),1):
    if any(s in l for s in ['ENABLE_BINARY','--jit','--binary','binary','ops_kernel','ascendc_bin','kernel_compile']):print(i,l[:300])
 pins=json.loads((P/'ORIGINAL_MATMUL_PINS.json').read_text());diff=[]
 for rel,h in pins.items():
  f=B/'source'/rel
  if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:diff.append(rel)
 print('SOURCE_144_CHECK',len(pins),'mismatches',diff)
