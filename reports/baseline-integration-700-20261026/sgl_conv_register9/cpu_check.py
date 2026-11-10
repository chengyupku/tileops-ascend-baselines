import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import ast, hashlib, importlib, json, pathlib, sys, types
root=pathlib.Path('/root/dyq/workspace')
sys.path.insert(0,str(root/'TileOPs'))
# Import only CPU-safe registry; no vendor module imports are allowed.
pkg=importlib.import_module('benchmarks.ascend_all_pairs.sgl')
spec=importlib.util.spec_from_file_location('integration700_sgl_conv_register9', root/'reports/baseline-integration-700-20261026/sgl_conv_register9/integration700_sgl_conv_register9.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
conv_ids=set(mod._i700_sgl_conv_register9_conv_ids) if hasattr(mod,'_i700_sgl_conv_register9_conv_ids') else {'resnet-1x1-float16','bottleneck-expand-float16','late-stage-1x1-bias-float16','classifier-1x1-float16','resnet-1x1-bfloat16','resnet-1x1-bfloat16','resnet-1x1-bias-bfloat16','classifier-1x1-bias-float16'}
chain_ids={'llama2-7b-b256-n1','llama-8b-b17-n8'}
class C:
    def __init__(self, op, cid): self.id=cid; self._call=types.SimpleNamespace(op=op)
# prevent original vendor imports while exercising factory construction
# This environment has no torch; factories only need the module object until run-time.
sys.modules.setdefault('torch', types.ModuleType('torch'))
api_stub=types.ModuleType('benchmarks.api')
api_stub.Implementation=type('Implementation', (), {'__init__':lambda self,run,*a,**kw:setattr(self,'run',run)})
sys.modules['benchmarks.api']=api_stub
import benchmarks.ascend_baselines as ab
old_local=ab.local_function
old_import=importlib.import_module
def fake_local(*args):
    return lambda *a, **kw: None
def fake_import(name,*args,**kwargs):
    if name=='sgl_kernel_npu':
        return types.SimpleNamespace(__file__=str(root/'ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/__init__.py'))
    return old_import(name,*args,**kwargs)
ab.local_function=fake_local
importlib.import_module=fake_import
rows=[]
try:
    for cid in sorted(conv_ids):
        cand=pkg.PROVIDERS['Conv2dFwdOp'](C('Conv2dFwdOp',cid))
        rows.append({'op':'Conv2dFwdOp','case_id':cid,'candidate_type':type(cand).__name__,'api':cand.api,'factory_called':True})
    for cid in sorted(chain_ids):
        cand=pkg.PROVIDERS['ChainSpeculativeSamplingFwdOp'](C('ChainSpeculativeSamplingFwdOp',cid))
        rows.append({'op':'ChainSpeculativeSamplingFwdOp','case_id':cid,'candidate_type':type(cand).__name__,'api':cand.api,'factory_called':True})
finally:
    ab.local_function=old_local
    importlib.import_module=old_import
# module and module.function forms
assert callable(mod.conv) and callable(mod.chain) and isinstance(mod.PROVIDERS,dict)
assert set(r['case_id'] for r in rows)==conv_ids|chain_ids
# static forbidden-import check on newly added files
for path in [root/'reports/baseline-integration-700-20261026/sgl_conv_register9/integration700_sgl_conv_register9.py',root/'TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py']:
    text=path.read_text(); assert 'torch_npu' not in text
    tree=ast.parse(text)
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in n.names]
            assert not any(x=='torch_npu' or x.startswith('torch_npu.') for x in names)
# source / provider hashes
out={'passed':True,'exact_keys':len(rows),'rows':rows,'module_forms':{'conv':True,'chain':True},'forbidden_import_scan':True,'torch_npu_imported':False,'adapter_sha256':hashlib.sha256((root/'reports/baseline-integration-700-20261026/sgl_conv_register9/integration700_sgl_conv_register9.py').read_bytes()).hexdigest(),'provider_sha256':hashlib.sha256((root/'TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py').read_bytes()).hexdigest()}
print(json.dumps(out,indent=2))
(root/'reports/baseline-integration-700-20261026/sgl_conv_register9/CPU_FACTORY_CHECK.json').write_text(json.dumps(out,indent=2)+'\n')
