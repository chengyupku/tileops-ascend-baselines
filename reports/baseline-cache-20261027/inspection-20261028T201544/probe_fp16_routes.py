"""Read-only CPU admission probe; never imports device runtimes or builds artifacts."""
import sys,importlib.abc,json,collections,time,types
from pathlib import Path
class NoDeviceImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','triton','tilelang','acl','tvm'):
   raise ImportError('CPU_ONLY_BLOCK:'+fullname)
sys.meta_path.insert(0,NoDeviceImports());sys.dont_write_bytecode=True
sys.path[:0]=[str(Path('TileOPs').resolve()),str(Path('tools').resolve())]
from run_baseline_cache import assigned_history,completed_ids,quarantine_key
from benchmarks.ascend_all_pairs import registry
r=Path('reports/baseline-cache-20261027');host=sys.argv[1];h=r/host;d=r/'inspection-20261028T201544'
tasks=[x for x in json.load(open(r/'PLAN.json'))['pairs'] if x['host']==host]
rows=[json.loads(x) for x in (h/'journal.jsonl').read_text().splitlines()];cache=[json.loads(x) for x in (h/'CACHE.jsonl').read_text().splitlines()]
latest,cached=assigned_history(tasks,rows,cache);q=json.load(open(h/'QUARANTINE.json'));done=completed_ids(latest,json.load(open(h/'RETRY.json')),q,cached)
ready=[t for t in tasks if t['id'] not in done and not quarantine_key(t,q) and t['key'][0] in ('ops-transformer','sgl-kernel-npu') and t['key'][1]=='GDNFwdOp' and t['case']['tensors']['q'][1]=='float16']
blocked=[];unknown=[]
def readonly(event,args):
 if event.startswith(('subprocess.','os.system','os.spawn','socket.')):raise RuntimeError('CPU_PROBE_BLOCK:'+event)
 if event in ('os.mkdir','os.rename','os.remove','os.rmdir'):raise RuntimeError('CPU_PROBE_BLOCK:'+event)
 if event=='open' and isinstance(args[2],int) and args[2] & (1|2|64|512|1024):raise RuntimeError('CPU_PROBE_BLOCK:write_open')
sys.addaudithook(readonly)
class RouteReached(Exception):pass
def trace(frame,event,arg):
 if event=='call' and ((frame.f_code.co_name=='recurrent_family' and frame.f_code.co_filename.endswith('ops_transformer/adapters.py')) or (frame.f_code.co_name=='_native' and frame.f_code.co_filename.endswith('sgl/linear.py'))):
  raise RouteReached('CPU_ROUTE_CONFIRMED:'+frame.f_code.co_filename+':'+frame.f_code.co_name)
for t in ready:
 c=t['case'];case=types.SimpleNamespace(id=t['key'][2],params=c['params'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=t['key'][1]),tensors=c['tensors']))
 try:
  sys.setprofile(trace);f=registry.candidate_factory(case,t['key'][0]);f();error='CPU factory returned; no numerical conclusion'
 except Exception as e:error=str(e)
 finally:sys.setprofile(None)
 if error.startswith('CPU_ROUTE_CONFIRMED:'):
  blocked.append({'id':t['id'],'key':t['key'],'reason':error})
 else:unknown.append({'id':t['id'],'key':t['key'],'observation':error[:150]})
print(json.dumps({'at':time.time(),'ready_examined':len(ready),'definite_cpu_admission_blocks':blocked,'other_not_classified':unknown,'no_device_runs':True},ensure_ascii=False))
