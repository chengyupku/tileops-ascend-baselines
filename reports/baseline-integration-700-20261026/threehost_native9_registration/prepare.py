import json,hashlib,subprocess,base64,shlex
from pathlib import Path
O=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration');S=O.parent/'threehost_native_completion9'
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
W='/opt/tileops-all-pairs/workspace'; R=W+'/reports/baseline-integration-700-20261026';old=R+'/threehost_native_completion9'
files=[W+'/TileOPs/benchmarks/'+n+'.py' for n in ['ascend_baselines','ascend_math_more','api']]+[R+'/transformer_dense/integration700_transformer_dense_fp8_bmm.py']+[old+'/arm_closure/'+n for n in ['bf16_moe_hadamard_native10/adapter.py','bf16_moe_hadamard_parent10/adapter.py','bf16_local_gemm18/binding.py']]
frozen={}
for mode in ['fp8','bf16arm2']:
 raw=(S/f'FROZEN_{mode}.json').read_bytes();(O/f'FROZEN_{mode}.json').write_bytes(raw);frozen.update(json.loads(raw)['sha256'])
runtime={p:h for p,h in frozen.items() if '/private_provider/' in p}
# Actual SDK/driver API dependencies only, no unrelated environment archive.
extra=['/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/'+n for n in ['libnnopbase.so','libascendcl.so','libacl_rt.so','libruntime.so']]+['/usr/local/Ascend/driver/lib64/driver/libascend_hal.so','/usr/local/Ascend/cann-9.1.0/set_env.sh']
script='''import json,hashlib,base64,platform\nfrom pathlib import Path\nfiles=%r\nruntime=%r\nextra=%r\nout={'sources':{},'runtime':{},'host':{'node':platform.node(),'machine':platform.machine(),'machine_id':Path('/etc/machine-id').read_text().strip()}}\nfor p in files:out['sources'][p]=base64.b64encode(Path(p).read_bytes()).decode()\nfor p in list(runtime)+extra:\n with open(p,'rb') as f:\n  h=hashlib.sha256()\n  for b in iter(lambda:f.read(1048576),b''):h.update(b)\n out['runtime'][p]=h.hexdigest()\nprint(json.dumps(out))\n'''%(files,runtime,extra)
data=json.loads(subprocess.check_output([ssh,'openlibing-DevEnv-661495','python3 -S -c '+shlex.quote(script)],text=True))
assert all(data['runtime'][p]==h for p,h in runtime.items())
sources={}
for p,encoded in data['sources'].items():
 raw=base64.b64decode(encoded);h=hashlib.sha256(raw).hexdigest()
 if h!=frozen[p]:
  snap=S/'snapshots'/p.removeprefix(W+'/'); raw=snap.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==frozen[p],p
 rel='closure/'+p.removeprefix(W+'/');dest=O/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);sources[p]={'file':rel,'sha256':h}
runtime.update({p:data['runtime'][p] for p in extra})
env=json.loads((S/'server1/llama-8b-prefill-bfloat16/result/RUNTIME.json').read_text())['environment']
env.update(ASCEND_HOME_PATH='/usr/local/Ascend/cann-9.1.0',ASCEND_OPP_PATH='/usr/local/Ascend/cann-9.1.0/opp',ASCEND_GLOBAL_LOG_LEVEL='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
(O/'PINS.json').write_text(json.dumps(dict(sources=sources,runtime=runtime,host=data['host'],environment=env,provenance='source/vendor hashes from original FROZEN files; SDK/driver pins read-only capture during registration'),indent=2)+'\n')
print('sources',len(sources),'runtime',len(runtime),'host',data['host'])
