"""CPU-only fresh execution directory and explicit original environment; no device imports."""
from pathlib import Path
import json,tempfile,os
P=Path(__file__).resolve().parent
(P/'runs').mkdir(exist_ok=True)
r=Path(tempfile.mkdtemp(prefix='run-',dir=P/'runs'));(r/'RUN_EXCLUSIVE').write_text('private single evaluation directory\n')
pins=json.loads((P/'PINS.json').read_text())
env=dict(pins['environment'],TILELANG_BMM_RUN_ROOT=str(r),TILELANG_BMM_GUARD_OWNER='baseline700-20261026/threehost_tilelang_bmm_registration3')
for n in ('ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TMPDIR','TMP','TEMP','ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR','TILELANG_CACHE_DIR'):
 (r/n).mkdir();env[n]=str(r/n)
(r/'environment.json').write_text(json.dumps(env,indent=2)+'\n');print(r)
