import importlib.util,sys,types,pathlib,json,hashlib,ast
p=pathlib.Path(__file__).resolve().parent;d=p.parents[2]/'TileOPs/benchmarks/ascend_all_pairs/fla'
name='benchmarks.ascend_all_pairs.fla';spec=importlib.util.spec_from_file_location(name,d/'__init__.py',submodule_search_locations=[str(d)])
m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
stub=types.ModuleType(name+'.integration700_fla_native');stub.build=lambda case:('integration700',case.id);sys.modules[stub.__name__]=stub
keys=set()
for f in p.glob('registration.*.json'):keys.update(tuple(k) for k in json.loads(f.read_text())['keys'])
for op,cid in keys:assert m.PROVIDERS[op](types.SimpleNamespace(id=cid))==('integration700',cid)
private=p/'integration700_fla_native.py';public=d/private.name
assert private.read_bytes()==public.read_bytes()
result=dict(device_free=True,registered_keys=sorted(keys),passed=True,private_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),public_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),torch_imported='torch' in sys.modules,torch_npu_imported='torch_npu' in sys.modules)
(p/'STATIC_CHECK.json').write_text(json.dumps(result,indent=2));print(result)
