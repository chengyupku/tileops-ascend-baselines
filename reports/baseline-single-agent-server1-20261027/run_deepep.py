"""Matched original DeepEP host package, canonical full checker, fixed server1."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

P=Path(__file__).resolve().parent
W=Path('/opt/tileops-all-pairs/workspace')
cid=sys.argv[1]
assert cid in {'ds-v3-mixed-bfloat16','llama-70b-cp-bfloat16'}
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert meta['owner']=='single-server1/deepep' and meta['phase']=='running' and meta['child']['pid']==os.getpgrp()
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
if os.environ.get('SINGLE_SERVER1_EP_REEXEC')!='1':
    env=dict(os.environ,SINGLE_SERVER1_EP_REEXEC='1',
             LD_LIBRARY_PATH=str(sdk/'lib64')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    os.execve(sys.executable,[sys.executable,'-I','-S','-B','/opt/tileops-all-pairs/control/private_python.py',str(Path(__file__).resolve()),cid],env)
out=P/('deepep-'+cid)
out.mkdir(exist_ok=False)
(out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
os.environ.update(ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),BISHENG_HOME=str(sdk/'tools/bisheng_compiler'),
                  DEEPEP_ALL_PAIRS_BUILD=str(W/'.cache/all-pairs/deepep/torch210-sdk920b2'))
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
for n in ['TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH']:
    d=out/n;d.mkdir();os.environ[n]=str(d)
import tempfile
tempfile.tempdir=os.environ['TMPDIR']
checker=W/'TileOPs/scripts/ascend_pair_check.py'
assert hashlib.sha256(checker.read_bytes()).hexdigest()=='3b9dd9dde43e5ff6c101496b714bd9d2710a6758264d4f851dd74da829e4bb9f'
sys.argv=[str(checker),'--op','PagedKVCacheGatherFwdOp','--case-id',cid,'--library','DeepEP-Ascend','--output',str(out/'checker.json')]
try:
    runpy.run_path(str(checker),run_name='__main__')
finally:
    (out/'loaded_maps.txt').write_text(Path('/proc/self/maps').read_text())
