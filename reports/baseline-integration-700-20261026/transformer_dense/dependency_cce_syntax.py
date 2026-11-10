from pathlib import Path
import subprocess,json,time
p=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense');c=p/'gemm/private_provider/op_impl/ai_core/tbe/recovery1217_gemm_transformer_impl/ascendc/common';sdk=Path('/usr/local/Ascend/cann')
source=p/'dependency_probe.cpp';source.write_text('#include "kernel_operator.h"\n#include "blaze/epilogue/block/block_epilogue_empty.h"\n')
cmd=[str(sdk/'aarch64-linux/bin/bisheng'),'-x','cce','--cce-aicore-arch=dav-c310','-std=c++17','-fsyntax-only',str(source)]
for x in [c/'tensor_api/include',c/'tensor_api',c,sdk/'include',sdk/'compiler/tikcpp/tikcfw',sdk/'asc/include',sdk/'asc/impl/c_api',sdk/'asc/impl/utils']:cmd+=['-I',str(x)]
t=time.monotonic()
with (p/'dependency_cce_syntax.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=90)
(p/'dependency_cce_syntax.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode,'elapsed_s':time.monotonic()-t,'NPU_used':False,'meaning':'CPU-only header syntax diagnostic, not original full target compile or numerical test'},indent=2))
