import sys, importlib.abc, json, collections, types, hashlib, time
from pathlib import Path
sys.dont_write_bytecode = True
class NoDevice(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','triton','tilelang','acl','tvm'):
            raise ImportError('CPU_DEVICE_BOUNDARY:' + fullname)
sys.meta_path.insert(0, NoDevice())
sys.path[:0] = [str(Path('TileOPs').resolve()), str(Path('tools').resolve())]
from benchmarks.ascend_all_pairs import registry
host=sys.argv[1]; tasks=json.loads(Path(sys.argv[2]).read_text()); results=[]
def audit(event,args):
    if event.startswith(('subprocess.','os.system','os.spawn','os.fork','socket.')) or event=='ctypes.dlopen':
        raise RuntimeError('CPU_PROBE_BLOCK:'+event)
    if event in ('os.mkdir','os.rename','os.remove','os.rmdir','os.link','os.symlink'):
        raise RuntimeError('CPU_PROBE_BLOCK:'+event)
    if event=='open' and isinstance(args[2],int) and args[2] & (1|2|64|512|1024):
        raise RuntimeError('CPU_PROBE_BLOCK:write_open')
sys.addaudithook(audit)
for t in tasks:
    c=t['case'];case=types.SimpleNamespace(id=t['key'][2],params=c['params'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=t['key'][1]),tensors=c['tensors']))
    try:
        f=registry.candidate_factory(case,t['key'][0]);f();error='CPU_FACTORY_RETURNED_NO_EXECUTION';kind='returned'
    except Exception as e:
        error=str(e)[:500];kind=type(e).__name__
    results.append({'id':t['id'],'key':t['key'],'kind':kind,'observation':error})
files=['registry.py','ops_math/__init__.py','ops_math/enabled_cases.json','ops_nn/__init__.py','ops_nn/accepted_cases.json','sgl/__init__.py','ops_transformer/__init__.py','fla/__init__.py','tla/__init__.py','deepgemm/__init__.py','catlass/vector.py','flashmla/build.py']
hashes={}
for name in files:
    p=Path('TileOPs/benchmarks/ascend_all_pairs')/name
    hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
checks={}
for name,path in {'catlass_mlir':'ascend_baselines/catlass/python/tla_dsl/catlass/_mlir','deepgemm_source':'ascend_baselines/DeepGEMM-Ascend','deepgemm_pkg':'ascend_baselines/DeepGEMM-Ascend/deep_gemm','opsmath_install':'.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math','opsnn_install3':'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'}.items():
    p=Path(path);checks[name]={'path':str(p.resolve()),'exists':p.exists()}
print(json.dumps({'host':host,'at':time.time(),'examined':len(results),'device_runs':0,'limitations':'A blocked device import is not a numerical pass. Earlier missing attribute/import can prevent reaching later checks. No native libraries loaded.','registry_hashes':hashes,'paths':checks,'counts':dict(collections.Counter(x['observation'] for x in results)),'rows':results},ensure_ascii=False))
