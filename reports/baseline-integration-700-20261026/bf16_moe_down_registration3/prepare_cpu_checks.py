from pathlib import Path
O=Path(__file__).resolve().parent
s=(O.parent/'bf16_moe_masked_registration1/cpu_checks.py').read_text().replace('integration700_bf16_moe_masked_registration1.py','integration700_bf16_moe_down_registration3.py').replace('BF16_MOE_MASKED_GUARD_OWNER','BF16_GEMM_GUARD_OWNER').replace('BF16_MOE_MASKED_RUN_ROOT','BF16_GEMM_RUN_ROOT')
a=s.index('realcall=next(');b=s.index('# Native dependencies must be private')
s=s[:a]+'''realcalls={c.case_id:c for r in load_workloads(op) for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id in {k[2] for k in M.CASE_PARAMS}}
assert len(realcalls)==3
realcall=realcalls['ds-v3-decode-down-bfloat16']
api_source=Path('/root/dyq/workspace/TileOPs/benchmarks/api.py').read_bytes()
tree=ast.parse(api_source)
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Case','_report_value')]
ns=dict(dataclasses=dataclasses,functools=functools,Any=typing.Any,Optional=typing.Optional,Callable=typing.Callable,Call=type(realcall),ADTValue=ADTValue)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-api-CPU-classes','exec'),ns)
cs=[ns['Case'](c.case_id,c,None) for c in realcalls.values()]
realcase=next(c for c in cs if c.id==realcall.case_id)
from benchmarks.ascend_baselines import UnsupportedCase

def public_reject(fn):
 try:fn()
 except UnsupportedCase as e:assert type(e) is UnsupportedCase;return
 raise AssertionError('Expected canonical public UnsupportedCase')

def case(k,p):
 raw=realcalls.get(k[2],realcall)
 return types.SimpleNamespace(id=k[2],params=p,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=k[1]),indices=dict(raw.indices),params=dict(raw.params),tensors=dict(raw.tensors),present=raw.present))
for c in cs:
 k=('ops-transformer',op,c.id)
 assert M.matches(c) and M._canonical(c.params)==M.CASE_PARAMS[k]
 assert M._canonical(c._call.indices)==M._canonical(M.CASE_INDICES[k])
 assert callable(M.make(c).implementation.run)
 assert not M.matches(c,'wrong-baseline')
 public_reject(lambda:M.make(c,'wrong-baseline'))
 for field in ('P','K','D','R','N','E'):
  bad=case(k,copy.deepcopy(c.params));bad._call.indices[field]='wrong'
  assert not M.matches(bad);public_reject(lambda:M.make(bad))
 for field,value in [('activation','silu_and_mul'),('out_dtype','float16'),('layout',{'contiguous':{'packing':'tight','metadata_kind':'physical_psum','alignment':1}})]:
  bad=case(k,copy.deepcopy(c.params));bad._call.params[field]=value
  assert not M.matches(bad);public_reject(lambda:M.make(bad))
 for where in ('params','indices'):
  for field,value in [('packing','padded'),('metadata_kind','counts'),('alignment',2),('padding',1)]:
   bad=case(k,copy.deepcopy(c.params));fields=dict(packing='tight',metadata_kind='physical_psum',alignment=1);fields[field]=value
   getattr(bad._call,where)['layout']=ADTValue('MGroupedLayout',{'contiguous':fields})
   assert not M.matches(bad);public_reject(lambda:M.make(bad))
 for name in ('a','b','output','layout_metadata'):
  for kind in ('shape','dtype'):
   bad=case(k,copy.deepcopy(c.params));shape,dtype=bad._call.tensors[name]
   bad._call.tensors[name]=(tuple([shape[0]+1,*shape[1:]]),dtype) if kind=='shape' else (shape,'float16')
   assert not M.matches(bad);public_reject(lambda:M.make(bad))
   bad=case(k,copy.deepcopy(c.params));shape,dtype=bad.params[name]
   bad.params[name]=(tuple([shape[0]+1,*shape[1:]]),dtype) if kind=='shape' else (shape,'float16')
   public_reject(lambda:M.make(bad))
 bad=case(k,dict(c.params,unexpected=True));public_reject(lambda:M.make(bad))
 bad=case(k,copy.deepcopy(c.params));bad._call.present=lambda n:False
 assert not M.matches(bad)
 mut=case(k,copy.deepcopy(c.params));cand=M.make(mut);frozen=M._freeze_case(mut)
 before=M._canonical(frozen.params)
 mut.params['layout']['contiguous']['alignment']=2
 public_reject(cand.implementation.run)
 assert M._canonical(frozen.params)==before and M.matches(frozen)
 assert frozen._call.tensors['a']==c._call.tensors['a']
 assert frozen._call.params['layout'].metadata_kind=='physical_psum'
 assert frozen._call.params['out_dtype']==c.params['out_dtype']
 for target in (frozen.params['layout']['contiguous'],frozen._call.params['layout'].fields,frozen._call.indices):
  try:target['alignment']=2
  except TypeError:pass
  else:raise AssertionError('mutable nested snapshot')
frozen=M._freeze_case(realcase)
passed('all 3 real manifest Calls and passive original Case classes; full params and per-key indices exact')
passed('all 3 shape/dtype/activation/out_dtype/layout/packing/padding/alignment/metadata fields rejected with public UnsupportedCase')
passed('immutable frozen snapshots unaffected by later caller mutations; mutated caller fails closed')
previous=[]
def old(c):previous.append(c);return 'old'
f=M.wrap(old)
for c in cs:
 unknown=case(('ops-transformer',op,'old-'+c.id),c.params);assert f(unknown)=='old'
assert len(previous)==3
other=case(('ops-transformer','OtherOp',realcase.id),realcase.params);assert f(other)=='old'
wrong=case(('ops-transformer',op,realcase.id),dict(realcase.params,stride=2));before=len(previous);public_reject(lambda:f(wrong));assert len(previous)==before
public_reject(lambda:M.wrap(None)(other))
passed('all 3 unknown keys delegate previous; known invalid keys never fall back; canonical public exception identity')
''' +s[b:]
s=s.replace('3D masked three-argument run','2D contiguous three-argument run')
# Block native library loading as well as Python device imports during CPU checks.
s=s.replace("builtins.__import__=protected", """builtins.__import__=protected
import ctypes
native_loads=[]
def no_native(*a,**kw):
 native_loads.append(str(a));raise AssertionError('Native library load forbidden')
ctypes.CDLL=no_native
ctypes.PyDLL=no_native""")
s=s.replace('assert not forbidden','assert not forbidden and not native_loads')
# Refresh heartbeat at every completed check, preserving output only in own dir.
s=s.replace('def passed(x):checks.append(x)',"""def passed(x):
 checks.append(x)
 import datetime
 (O/'HEARTBEAT.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=x,npu_runs=0)))""")
(O/'cpu_checks.py').write_text(s)
