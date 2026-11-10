import json,hashlib,subprocess,base64,shlex,datetime
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent;S=B/'threehost_bmm_backend_full1_r2'
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm_backend_full1_r2'
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
def dump(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
def hb(s):dump('HEARTBEAT.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=s,accepted_added=0))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
hb('reviewing_seal_and_fetching_original_closure_cpu_only')
frozen=json.loads((S/'server1/FROZEN.json').read_text())['sha256']
sealed=json.loads((S/'server1/SEALED_HASHES.json').read_text())
print('seal type',type(sealed).__name__,'count',len(sealed),'sample keys',list(sealed)[:2])
files=[R+'/'+x for x in ['adapter.py','binding.py','TileOPs/benchmarks/api.py','TileOPs/benchmarks/ascend_baselines.py','TileOPs/benchmarks/ascend_math_more.py','TileOPs/scripts/ascend_pair_check.py','TileOPs/workloads/gemm.py','TileOPs/workloads/numerics.py']]
runtime={p:h for p,h in frozen.items() if '/private_provider/' in p}
extra=['/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/'+n for n in ['libnnopbase.so','libascendcl.so','libacl_rt.so','libruntime.so']]+['/usr/local/Ascend/driver/lib64/driver/libascend_hal.so','/usr/local/Ascend/cann-9.1.0/set_env.sh']
loaded=json.loads((S/'server1/square-4k-bfloat16/EXECUTION.json').read_text())['loaded_files']
extra += [p for p in loaded if '/site-packages/torch/' in p or '/site-packages/torch_npu/' in p]
extra += ['/opt/tileops-ci/runtime/bin/python','/opt/tileops-all-pairs/control/private_python.py']
script='''import json,hashlib,base64,platform\nfrom pathlib import Path\nfiles=%r\nruntime=%r\nextra=%r\nout={'sources':{},'runtime':{},'host':{'node':platform.node(),'machine':platform.machine()}}\nfor p in files:out['sources'][p]=base64.b64encode(Path(p).read_bytes()).decode()\nfor p in list(runtime)+extra:\n with open(p,'rb') as f:\n  h=hashlib.sha256()\n  for b in iter(lambda:f.read(1048576),b''):h.update(b)\n out['runtime'][p]=h.hexdigest()\nprint(json.dumps(out))\n'''%(files,runtime,extra)
r=subprocess.run([SSH,'openlibing-DevEnv-661495','python3 -S -'],input=script,capture_output=True,text=True,timeout=120)
assert r.returncode==0,r.stderr
d=json.loads(r.stdout);assert all(d['runtime'][p]==h for p,h in runtime.items())
sources={}
for p,s in d['sources'].items():
 raw=base64.b64decode(s);h=hashlib.sha256(raw).hexdigest();assert h==frozen[p],p
 rel='closure/'+p.removeprefix(R+'/');dest=O/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);sources[p]=dict(file=rel,sha256=h)
env=json.loads((S/'server1/square-4k-bfloat16/RUNTIME.json').read_text())['environment']
env.update(ASCEND_HOME_PATH='/usr/local/Ascend/cann-9.1.0',ASCEND_OPP_PATH='/usr/local/Ascend/cann-9.1.0/opp',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
dump('PINS.json',dict(sources=sources,runtime=d['runtime'],host=d['host'],environment=env,backend_variant='torch.backends.mkldnn.enabled=False',provenance='original sealed FROZEN source and vendor hashes rechecked CPU-only on server1; SDK hashes freshly captured'))
(O/'FROZEN.json').write_bytes((S/'server1/FROZEN.json').read_bytes())
print('sources',len(sources),'runtime',len(d['runtime']),'all original frozen hashes match')
