from pathlib import Path
import json,fcntl,hashlib,datetime,ast,difflib
W=Path(__file__).resolve().parent;R=W.parents[2];P=R/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer';name='integration700_transformer_linear_gdn'
success=[]
for i in json.loads((W/'CHUNK_CASES.json').read_text()):
 f=W/i/'checker.json';g=W/i/'GUARD_RECEIPT.json'
 if f.exists() and g.exists():
  c=json.loads(f.read_text());guard=json.loads(g.read_text());v=c.get('verification',{})
  if c.get('status')=='verified' and guard['returncode']==0 and v.get('checked_outputs')==v.get('total_outputs')==2:success.append(i)
assert success
module='''"""Exact successful GDN keys; unchanged original complete ACLNN binding."""
from pathlib import Path
import json, importlib.util, sys
PRIVATE=Path(%r)
def build(case):
    package=PRIVATE/'frozen_v1/chunk_package'
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    transport=load('_integration700_transformer_linear_transport',package/'direct_acl.py')
    old=sys.modules.get('direct_acl');sys.modules['direct_acl']=transport
    try: binding=load('_integration700_transformer_linear_binding',package/'binding.py')
    finally:
        if old is None:sys.modules.pop('direct_acl',None)
        else:sys.modules['direct_acl']=old
    return binding.build(case)
def wrap(previous):
    def dispatch(case):
        keys=json.loads((PRIVATE/'REGISTERED.json').read_text())
        if case.id in keys:return build(case)
        if previous is None:raise KeyError(case.id)
        return previous(case)
    return dispatch
'''%str(W)
ast.parse(module)
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 dest=P/'__init__.py';before=dest.read_text();tag='# integration700_transformer_linear exact GDN keys'
 if tag not in before:
  (W/'provider_init.backup.py').write_text(before)
  after=before+'\n'+tag+'\nfrom .'+name+' import wrap as _integration700_transformer_linear_wrap\nPROVIDERS["GDNFwdOp"] = _integration700_transformer_linear_wrap(PROVIDERS.get("GDNFwdOp"))\n'
  ast.parse(after);dest.write_text(after);(W/'provider.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(dest),tofile=str(dest))))
 (P/(name+'.py')).write_text(module);(W/(name+'.py')).write_text(module)
 t=W/'REGISTERED.tmp';t.write_text(json.dumps(sorted(success),indent=2));t.replace(W/'REGISTERED.json')
 (W/'REGISTRATION.json').write_text(json.dumps({'keys':success,'factory':f'benchmarks.ascend_all_pairs.ops_transformer.{name}.build','module_sha256':hashlib.sha256(module.encode()).hexdigest(),'provider_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'device_free_check':'AST parsed; exact-key fallback wrapper; unchanged private numeric binding'},indent=2))
print('registered',len(success))
