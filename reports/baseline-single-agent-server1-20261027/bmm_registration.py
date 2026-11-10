"""Exact-case registration; lazy device imports and original baseline dispatch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

P = Path(__file__).resolve().parent
W = Path('/opt/tileops-all-pairs/workspace')

def install(providers):
    previous = providers.get('BmmFwdOp')
    contracts = json.loads((P/'BMM_CONTRACTS.json').read_text())
    def provider(case):
        if case.id not in contracts:
            return previous(case)
        expected = contracts[case.id]
        plain = lambda x: json.loads(json.dumps(x))
        if plain(case.params) != expected['parameters'] or plain(case._call.tensors) != expected['declared_tensors']:
            raise ValueError('Exact verified Bmm contract changed')
        metadata = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
        assert metadata['phase'] == 'running' and metadata['child']['pid'] == os.getpgrp()
        assert not any(n in sys.modules for n in ('tilelang','deepgemm','tvm')), 'Fresh process required'
        pins = json.loads((P/'BMM_SOURCE_PINS.json').read_text()) | json.loads((P/'RUNTIME_PINS.json').read_text())
        for name, sha in pins.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
        for name in ['TILELANG_LIBRARY_PATH','TVM_LIBRARY_PATH','TVM_HOME','TVM_IMPORT_PYTHON_PATH','TL_ROOT','TL_PLATFORM','ASCEND_NPU_ARCH']:
            os.environ.pop(name, None)
        root = P/'registered-runs'
        root.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix=case.id+'-', dir=root))
        for name in ['TILELANG_CACHE_DIR','TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH']:
            d=out/name; d.mkdir(); os.environ[name]=str(d)
        tempfile.tempdir=os.environ['TMPDIR']
        sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920-release-headers'
        os.environ.update(BISHENG_HOME=str(sdk/'tools/bisheng_compiler'),ASCEND_HOME_PATH=str(sdk),
                          ASCEND_TOOLKIT_HOME=str(sdk),CPATH=':'.join(str(sdk/x) for x in ['asc/include','asc','include']))
        if '/opt/tileops-ci/runtime/lib/python3.12/site-packages' not in sys.path:
            sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
        spec=importlib.util.spec_from_file_location('_server1_complete_bmm',P/'bmm_adapter.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from dataclasses import replace
        candidate=module.factory(case,out)
        from tilelang.jit.kernel import JITKernel
        import torch
        original=JITKernel.__call__
        count=0
        def trace(kernel,*args,**kwargs):
            nonlocal count
            count+=1
            artifact=out/f'{count}.runtime-module.so'
            kernel.adapter.rt_mod.export_library(str(artifact))
            record={'target':str(kernel.target),'backend':kernel.execution_backend,
                    'exported_actual_runtime_module':str(artifact),
                    'sha256_before':hashlib.sha256(artifact.read_bytes()).hexdigest()}
            (out/f'{count}.dispatch.json').write_text(json.dumps(record,indent=2))
            result=original(kernel,*args,**kwargs)
            torch.npu.synchronize()
            record.update(synchronized=True,sha256_after=hashlib.sha256(artifact.read_bytes()).hexdigest())
            (out/f'{count}.dispatch.json').write_text(json.dumps(record,indent=2))
            (out/f'{count}.loaded_maps.txt').write_text(Path('/proc/self/maps').read_text())
            return result
        JITKernel.__call__=trace
        def cleanup():
            JITKernel.__call__=original
            changed=[n for n,sha in pins.items() if hashlib.sha256(Path(n).read_bytes()).hexdigest()!=sha]
            (out/'SOURCE_AFTER.json').write_text(json.dumps({'changed':changed},indent=2))
            assert not changed
        (out/'PRE_USE_IDENTITY.json').write_text(json.dumps({'pins':pins,'guard':metadata},indent=2))
        return replace(candidate,source_sha256=candidate.source_sha256|pins,cleanup=cleanup)
    providers['BmmFwdOp']=provider
