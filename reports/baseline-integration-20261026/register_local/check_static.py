import ast,hashlib,importlib,json,os,sys,types
from pathlib import Path
W=Path('/root/dyq/workspace');R=W/'reports/baseline-integration-20261026/register_local';P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn'
sys.dont_write_bytecode=True
class NoDevice:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','torch_npu','acl','tilelang'}:raise AssertionError('Device import forbidden: '+fullname)
sys.meta_path.insert(0,NoDevice());sys.path.insert(0,str(W/'TileOPs'))
b=types.ModuleType('benchmarks.ascend_baselines');b.WORKSPACE=W;b.UnsupportedCase=type('UnsupportedCase',(Exception,),{});sys.modules[b.__name__]=b
for row in json.loads((R/'CHANGED_FILES.json').read_text()):
 p=Path(row['path'])
 if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
pkg=importlib.import_module('benchmarks.ascend_all_pairs.ops_nn');v=importlib.import_module(pkg.__name__+'.verified_runtime');reg=importlib.import_module('benchmarks.ascend_all_pairs.registry')
records=list(v.RECORDS.values());assert len(records)==16
ns=types.SimpleNamespace
case=lambda op,cid:ns(_call=ns(signature=ns(name=op)),id=cid,params={})
# Compare old provider map to ensure every historical provider remains.
old={'__file__':str(P/'__init__.py'),'__name__':pkg.__name__,'__package__':pkg.__name__}
exec(compile((R/'backups/__init__.py').read_text(),str(P/'__init__.py'),'exec'),old)
assert set(old['PROVIDERS'])<=set(pkg.PROVIDERS)
original_make=v.make
v.make=lambda c:('selected',c._call.signature.name,c.id)
for r in records:
 _,op,cid=r['key'];assert reg.candidate_factory(case(op,cid),'ops-nn')()==('selected',op,cid)
# Sentinel fallback tests include failed InstanceNorm and an arbitrary case for every op.
for op in {k[0] for k in v.RECORDS}:
 sent=object();providers={op:lambda c:sent};v.install(providers);assert providers[op](case(op,'unknown-case')) is sent
assert pkg.PROVIDERS['InstanceNormFwdOp'].__name__==old['PROVIDERS']['InstanceNormFwdOp'].__name__
assert not any(k[0] in {'InstanceNormFwdOp','SumFwdOp'} for k in v.RECORDS)
v.make=original_make
# Confirm every lazy concrete factory dispatch without loading its device dependencies.
for r in records:
 _,op,cid=r['key'];module,function=r['factory'].split(':');fullname=pkg.__name__+'.verified_20261026.'+module
 mock=types.ModuleType(fullname);setattr(mock,function,lambda c:('factory',c.id));sys.modules[fullname]=mock
 os.environ['OPS_NN_VENDOR']=r['vendor'];os.environ['ASCEND_CUSTOM_OPP_PATH']=r['vendor']
 # Hash check once per record below; factory dispatch stubs only validation here.
 real_validate=v.validate;v.validate=lambda r:None
 assert reg.candidate_factory(case(op,cid),'ops-nn')()==('factory',cid)
 v.validate=real_validate
# Full receipt/source dependency hash check, cached reads for overlapping vendor inventories.
pins={};checks=[]
for r in records:
 c=json.loads(Path(r['checker']).read_text());assert c['status']=='verified' and c['inputs_preserved']
 q=c['verification'];assert q['checked_outputs']==q['total_outputs']>0
 for name,h in c['source_sha256'].items():
  path=W/name
  if path not in pins:pins[path]=hashlib.sha256(path.read_bytes()).hexdigest()
  assert pins[path]==h,(str(path),h,pins[path])
 checks.append({'key':r['key'],'checked_outputs':q['checked_outputs'],'total_outputs':q['total_outputs'],'reused':True})
# Relocations differ only in the two documented import statements.
for item in json.loads((R/'RELOCATION.json').read_text()):
 s=Path(item['source']).read_text()
 for a,b in item['changes']:s=s.replace(a,b)
 assert s==Path(item['destination']).read_text()
 assert ast.dump(ast.parse(s))==ast.dump(ast.parse(Path(item['destination']).read_text()))
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
result={'registered_count':16,'verified_receipts_reused_count':16,'unintegrated_keys':[],'historical_provider_count':len(old['PROVIDERS']),'historical_providers_retained':True,'unique_receipt_dependency_files_rehashed':len(pins),'static_compile':'passed','registry_and_lazy_dispatch_mock':'passed','negative_scope_and_fallback':'passed','device_imports':0,'new_numerical_runs':0,'receipts':checks}
(R/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='receipts'},indent=2))
