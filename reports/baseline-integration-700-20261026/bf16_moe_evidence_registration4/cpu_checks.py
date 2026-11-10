import builtins, importlib.util, json, os, sys, tempfile, types
from pathlib import Path
O=Path(__file__).resolve().parent
orig=builtins.__import__; forbidden=[]
def imp(name,*a,**kw):
    if name.split('.')[0] in ('torch','torch_npu','acl'):
        forbidden.append(name); raise AssertionError('forbidden device import')
    return orig(name,*a,**kw)
builtins.__import__=imp
spec=importlib.util.spec_from_file_location('reg4',O/'integration700_bf16_moe_hadamard_registration4.py'); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
checks=[]
def case(k,p): return types.SimpleNamespace(id=k[2],params=p,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=k[1])))
cs=[case(k,v) for k,v in M.CASE_PARAMS.items()]
for c in cs:
    assert M.matches(c); cand=M.make(c); assert callable(cand.implementation.run)
    bad=case(('ops-transformer',c._call.signature.name,c.id),dict(c.params,unexpected=True)); assert not M.matches(bad)
checks.append('4 exact canonical keys construct lazily; changed params rejected')
seen=[]
def prev(c): seen.append(c); return 'fallback'
wrapped=M.wrap(prev); unknown=case(('ops-transformer','OtherOp','unknown'),{}); assert wrapped(unknown)=='fallback' and len(seen)==1
checks.append('unknown route preserves previous fallback')
poison=types.ModuleType('adapter'); old=sys.modules.get('adapter'); sys.modules['adapter']=poison
try:
    private=M._load_private(); assert private.__name__.startswith('_bf16_moe_reg4_private_'); assert sys.modules['adapter'] is poison
finally:
    if old is None: sys.modules.pop('adapter',None)
    else: sys.modules['adapter']=old
checks.append('dependency bytes loaded privately without sys.modules pollution')
with tempfile.TemporaryDirectory() as td:
    M.ENV # immutable mapping proxy
    oldroot=os.environ.get('BF16_MOE_REG4_RUN_ROOT'); os.environ['BF16_MOE_REG4_RUN_ROOT']=td
    os.environ.pop('BF16_MOE_REG4_GUARD',None)
    try:
        try: M.make(cs[0]).implementation.run()
        except RuntimeError: pass
        else: raise AssertionError('guard must reject')
        os.environ['BF16_MOE_REG4_GUARD']='1'; cand=M.make(cs[0]); assert (Path(td).exists());
        try: cand.implementation.run()
        except RuntimeError as e: assert 'CPU preparation' in str(e)
    finally:
        if oldroot is None: os.environ.pop('BF16_MOE_REG4_RUN_ROOT',None)
        else: os.environ['BF16_MOE_REG4_RUN_ROOT']=oldroot
checks.append('guard admission precedes execution and CPU run fails closed without device import')
assert not forbidden
result={'passed':True,'checks':checks,'device_import_attempts':forbidden,'npu_runs':0,'numerical_runs':0,'native_library_loads':0,'mock_numeric_claims':False}
(O/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
