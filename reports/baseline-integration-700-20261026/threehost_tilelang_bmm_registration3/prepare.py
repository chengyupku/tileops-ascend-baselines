import ast,hashlib,json,pathlib,shutil
O=pathlib.Path(__file__).resolve().parent
E=O.parent/'threehost_tla_other54_r2/server1_evidence'
P=O.parent/'threehost_native9_registration'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
(O/'closure').mkdir(exist_ok=True)
scope=json.loads((O/'SCOPE.json').read_text());sources={};runtime={};provenance={};receipts=[]
W='/opt/tileops-all-pairs/workspace';A=W+'/reports/baseline-integration-700-20261026/threehost_tla_other54_r2/tilelang_adapter.py'
def source(path,raw,h):
 assert sha(raw)==h,path
 f='closure/'+h; (O/f).write_bytes(raw);sources[path]={'file':f,'sha256':h}
for r in scope:
 d=E/'runs'/('_'.join((r['baseline'],r['operator'],r['case_id'])))
 n=json.loads((d/'native.json').read_text())
 for path,h in n['hashes'].items():
  if path==A:source(path,(E/'tilelang_adapter.py').read_bytes(),h)
  elif path.startswith(W+'/tilelang/examples/'):source(path,(E/'source_freeze'/h).read_bytes(),h)
  else:
   assert path not in runtime or runtime[path]==h
   runtime[path]=h;provenance[path]='successful native.json hash'
 receipts.append({'key':[r['baseline'],r['operator'],r['case_id']],'directory':str(d),'checker_sha256':sha((d/'checker.json').read_bytes()),'native_sha256':sha((d/'native.json').read_bytes()),'guard_sha256':sha((d/'GUARD_RUNNING.json').read_bytes())})
for name in ['api.py','ascend_baselines.py']:
 path=W+'/TileOPs/benchmarks/'+name
 entry=next(x for x in json.loads((E/'FREEZE.json').read_text()) if x['original']==path)
 source(path,(E/'frozen/benchmarks'/name).read_bytes(),entry['sha256'])
dep=json.loads((E/'DEPENDENCY_ARTIFACT_HASHES.json').read_text())
for path,item in dep['files'].items():
 if path in sources:continue
 if path in runtime:assert runtime[path]==item['sha256']
 else:runtime[path]=item['sha256'];provenance[path]='post-run dependency inventory; historical association requires parent audit'
pins={'sources':sources,'runtime':runtime,'runtime_provenance':provenance,'receipts':receipts,'host':{'node':'openlibing-DevEnv-661495','machine':'aarch64'},'environment':{},'historical_sdk_audit':'pending parent; post-run inventory is not execution-time proof','accepted_added':0}
sdk=W+'/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920-release-headers'
pins['environment']={'BISHENG_HOME':sdk+'/tools/bisheng_compiler','ASCEND_HOME_PATH':sdk,'ASCEND_TOOLKIT_HOME':sdk,'CPATH':':'.join(sdk+'/'+x for x in ['asc/include','asc','include'])}
save('PINS.json',pins)
src=(P/'registration_parent.py').read_text()
src=src.replace('threehost_native9_registration','threehost_tilelang_registration3').replace('NATIVE9_','TILELANG3_').replace('native9','tilelang3')
start=src.index('DATA_HASHES =');end=src.index('\n\n',start);src=src[:start]+'DATA_HASHES = {}'+src[end:]
# Replace private loader: source bytes remain exact and only dynamically requested pinned examples may execute.
a=src.index('def private_load(');b=src.index('\ndef _proc(',a)
src=src[:a]+'''def private_load(data):
    import importlib, importlib.util
    original_import=builtins.__import__
    def spec(name,path):
        path=str(path)
        if path not in data:raise RuntimeError('Unpinned source import: '+path)
        class Loader:
            def create_module(self,spec):return None
            def exec_module(self,mod):
                mod.__file__=path
                exec(compile(data[path],path,'exec'),mod.__dict__)
        return importlib.util.spec_from_loader(name,Loader())
    bridge=types.SimpleNamespace(util=types.SimpleNamespace(spec_from_file_location=spec,module_from_spec=importlib.util.module_from_spec))
    def private_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name in ('importlib','importlib.util'):return bridge
        return original_import(name,globals,locals,fromlist,level)
    def load(path):
        mod=types.ModuleType('_tilelang3_original');mod.__file__=path
        mod.__dict__['__builtins__']=dict(vars(builtins),__import__=private_import)
        exec(compile(data[path],path,'exec'),mod.__dict__)
        return mod
    return load

''' +src[b:]
src=src.replace("'ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR')","'ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR','TILELANG_CACHE_DIR')")
src=src.replace("for name in ('LD_PRELOAD','PYTHONHOME'):","for name in ('LD_PRELOAD','PYTHONHOME','TILELANG_LIBRARY_PATH','TVM_LIBRARY_PATH','TVM_HOME','TVM_IMPORT_PYTHON_PATH','TL_ROOT','TL_PLATFORM','ASCEND_NPU_ARCH'):")
src=src.replace('names=list(case._call.signature.inputs)','names=[n for n in case._call.signature.inputs if case._call.present(n)]')
src=src.replace('frozen=copy.deepcopy(case);data=snapshot()','frozen=types.SimpleNamespace(_call=copy.deepcopy(case._call),id=case.id,params=copy.deepcopy(case.params));data=snapshot()')
src=src.replace("state['inner']=(mod.fp8_bmm if key(frozen)[0]=='BmmFP8FwdOp' else mod.make_candidate)(frozen)","frozen.inputs=args\n                out=Path(tempfile.mkdtemp(prefix='candidate-',dir=env[1]))\n                state['inner']=mod.factory(frozen,key(frozen)[0],out)")
src=src.replace("mod=load(FP8 if key(frozen)[0]=='BmmFP8FwdOp' else BF16)","mod=load(ADAPTER)")
src=src.replace("'ops-transformer::aclnnGroupedMatmulV5'","'tilelang::original-ascend'").replace('Exact tilelang3 contract; lazy original complete ARM9.1 factory; private pinned closure.','Exact tilelang contract; unchanged original adapter; lazy guarded original TileLang 0.1.15 Ascend.')
# Check runtime environment before tensor validation; no accidental runtime import on wrong host.
src=src.replace('validate_inputs(frozen,args,kwargs)','environment()\n            validate_inputs(frozen,args,kwargs)')
src=src.replace("loaded_libraries()\n            return result","loaded_libraries()\n            if require_guard()!=guard or environment()!=env:raise RuntimeError('Execution lifetime changed')\n            return result")
src=src.replace("if inner is not None and inner.cleanup is not None:inner.cleanup()","if inner is not None:\n            import torch\n            torch.npu.synchronize()\n            if inner.cleanup is not None:inner.cleanup()")
src += '\nADAPTER = '+repr(A)+'\n'
(O/'registration.py').write_text(src)
# Build complete actual manifest calls without importing API/torch.
b=(P/'build_contracts.py').read_text();b=b[:b.index("if __name__=='__main__':")]
b=b.replace(str(P),str(O)).replace("api=O/'closure/TileOPs/benchmarks/api.py'","api=O/"+repr(sources[W+'/TileOPs/benchmarks/api.py']['file']))
b=b.replace("wanted={(r['op'],r['case_id']) for r in json.loads((O.parent/'threehost_native_completion9/CONTRACTS.json').read_text())}","wanted={(r['operator'],r['case_id']) for r in json.loads((O/'SCOPE.json').read_text())}").replace('assert len(out)==9','assert len(out)==3')
b+='\nif __name__=="__main__":\n import registration\n (O/"CONTRACTS.json").write_text(json.dumps([registration.contract(c) for c in cases()],indent=2)+"\\n")\n'
(O/'build_contracts.py').write_text(b)
