import pathlib,json,sys,sysconfig,shutil,os,subprocess
W=pathlib.Path('/opt/tileops-all-pairs/workspace');B=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build';P=W/'reports/baseline-integration-700-20261026/threehost_nn11'
paths=[B/'source/matmul/quant_batch_matmul_v3/op_api/aclnn_quant_matmul_v4.h',B/'source/matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3.cpp',P/'sources/ascend_baselines/ops-nn/matmul/quant_batch_matmul_v3/op_api/aclnn_quant_matmul_v4.h',P/'sources/ascend_baselines/ops-nn/matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3.cpp',P/'snapshot/TileOPs/scripts/ascend_pair_check.py',pathlib.Path('/opt/tileops-all-pairs/bin/allpairs-python')]
d={'python':sys.executable,'abi':sysconfig.get_config_var('SOABI'),'env':{k:os.environ.get(k) for k in ['ASCEND_HOME_PATH','ASCEND_OPP_PATH','PATH','LD_LIBRARY_PATH']},'paths':{str(p):p.exists() for p in paths},'compiler':{k:shutil.which(k) for k in ['bisheng','ccec','cmake','g++']},'build_config':{},'sdk_templates':[]}
f=B/'build/CMakeCache.txt'
if f.exists():d['build_config']=[x for x in f.read_text().splitlines() if any(k in x for k in ['ASCEND_CANN_PACKAGE_PATH:','ASCEND_COMPUTE_UNIT:','OPS_NAME:','CMAKE_CXX_COMPILER:','ENABLE_JIT:'])]
for b in [pathlib.Path('/usr/local/Ascend/ascend-toolkit/latest'),pathlib.Path('/usr/local/Ascend/cann')]:
 if b.exists():
  d['sdk_templates'] += [str(x) for x in b.glob('**/ascendc/customize/cmake/ascendc.cmake')]
print(json.dumps(d))
