"""Run the unchanged full checker with a private original-API provider."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

P = Path(__file__).resolve().parent
W = Path('/opt/tileops-all-pairs/workspace')
case_id = sys.argv[1]
assert case_id in {'mha-decode-pv-bfloat16', 'moe-prefill-bfloat16'}
guard = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert guard['phase'] == 'running' and guard['child']['pid'] == os.getpgrp()
assert guard['owner'] == 'single-server1/bmm'
out = P / ('bmm_' + case_id)
out.mkdir(exist_ok=False)
(out / 'GUARD_RUNNING.json').write_text(json.dumps(guard, indent=2))
for name in ['TILELANG_LIBRARY_PATH','TVM_LIBRARY_PATH','TVM_HOME','TVM_IMPORT_PYTHON_PATH','TL_ROOT','TL_PLATFORM','ASCEND_NPU_ARCH']:
    os.environ.pop(name, None)
for name in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH']:
    path = out / name
    path.mkdir()
    os.environ[name] = str(path)
import tempfile
tempfile.tempdir = os.environ['TMPDIR']
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
sdk = W / '.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920-release-headers'
assert sdk.is_dir()
os.environ.update(BISHENG_HOME=str(sdk/'tools/bisheng_compiler'), ASCEND_HOME_PATH=str(sdk),
                  ASCEND_TOOLKIT_HOME=str(sdk), CPATH=':'.join(str(sdk/x) for x in ['asc/include','asc','include']))
pins = json.loads((P / 'BMM_SOURCE_PINS.json').read_text())
for name, sha in pins.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
import torch
import tilelang
from tilelang.jit.kernel import JITKernel
original_call = JITKernel.__call__
sequence = 0
def traced(kernel, *args, **kwargs):
    global sequence
    sequence += 1
    record = {'sequence': sequence, 'target': str(kernel.target), 'backend': kernel.execution_backend,
              'started': time.time(), 'adapter': str(type(kernel.adapter))}
    source = kernel.get_kernel_source()
    (out / f'{sequence}.kernel.txt').write_text(source)
    record['kernel_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    (out / f'{sequence}.dispatch.json').write_text(json.dumps(record, indent=2))
    result = original_call(kernel, *args, **kwargs)
    torch.npu.synchronize()
    native = []
    for name in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if str(out) in line and '.so' in line}):
        path = Path(name)
        if path.is_file():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            dest = out / 'native_artifacts' / sha
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists(): dest.write_bytes(path.read_bytes())
            native.append({'path': name, 'sha256': sha, 'inode': path.stat().st_ino})
    record.update(completed=time.time(), native=native)
    (out / f'{sequence}.dispatch.json').write_text(json.dumps(record, indent=2))
    return result
JITKernel.__call__ = traced
from benchmarks.ascend_all_pairs import tilelang as provider
from bmm_adapter import factory
provider.PROVIDERS['BmmFwdOp'] = lambda case: factory(case, out)
checker = W / 'TileOPs/scripts/ascend_pair_check.py'
assert hashlib.sha256(checker.read_bytes()).hexdigest() == '3b9dd9dde43e5ff6c101496b714bd9d2710a6758264d4f851dd74da829e4bb9f'
sys.argv = [str(checker), '--op', 'BmmFwdOp', '--case-id', case_id,
            '--library', 'tilelang', '--output', str(out/'checker.json')]
try:
    runpy.run_path(str(checker), run_name='__main__')
finally:
    changed = [name for name, sha in pins.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha]
    (out/'SOURCE_AFTER.json').write_text(json.dumps({'changed': changed, 'runtime': tilelang.__file__}, indent=2))
