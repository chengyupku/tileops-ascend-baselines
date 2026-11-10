import ast,hashlib,json,pathlib,sys,types
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];sys.path.insert(0,str(W/'TileOPs'))
from benchmarks.ascend_all_pairs import ops_nn
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
results=[]
for a in ['w4graph01','adagraph01','adagraph02','adagraph03']:
 r=json.loads((P/a/'REGISTRATION.json').read_text());b,op,cid=r['key'];module=pathlib.Path(r['module']);assert hashlib.sha256(module.read_bytes()).hexdigest()==r['sha256'];ast.parse(module.read_text())
 sentinel=object();setattr(ops_nn,'_nn_math_'+a,lambda c, value=sentinel:value)
 case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 assert ops_nn.PROVIDERS[op](case) is sentinel
 results.append(dict(key=r['key'],source_sha256=r['sha256'],route_passed=True))
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
(P/'DEVICE_FREE_CHECKS.json').write_text(json.dumps(dict(rows=results,device_imports=False),indent=2))
print('4 exact formal routes passed; no device runtime imported')
