import json,subprocess,pathlib,platform,time,hashlib
p=pathlib.Path(__file__).resolve().parent
w=pathlib.Path('/root/dyq/workspace') if platform.machine()=='x86_64' else pathlib.Path('/opt/tileops-all-pairs/workspace')
def run(c):
 try:
  r=subprocess.run(c,text=True,capture_output=True,timeout=20);return dict(argv=c,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as e:return dict(error=str(e))
r=dict(at=time.time(),machine=platform.machine(),hostname=platform.node(),guard=run(['npu-guard','status','--json']),sdk=[],runtimes=[])
for root in [pathlib.Path('/usr/local/Ascend'),w/'.cache/all-pairs/remote_native_deepgemm_deepep']:
 for sdk in root.glob('*'):
  b=sdk/'tools/bisheng_compiler/bin/bisheng'
  if b.exists():r['sdk'].append(dict(path=str(sdk),compiler=run([str(b),'--version'])))
for f in [w/'ascend_baselines/tilelang-ascend/tilelang/jit/adapter/libgen.py',w/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package/tilelang/jit/adapter/libgen.py',pathlib.Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages/tilelang/jit/adapter/libgen.py')]:
 if f.exists():r['runtimes'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),arch_lines=[l for l in f.read_text().splitlines() if any(x in l for x in ['npu-arch','aicore-arch','target ==','platform'])]))
wrapper=pathlib.Path('/opt/tileops-all-pairs/bin/allpairs-python');r['wrapper']=wrapper.read_text() if wrapper.exists() else None
if platform.machine()!='x86_64':r['private_python_cpu']=run(['/opt/tileops-ci/runtime/bin/python','-I','-S','-B','/opt/tileops-all-pairs/control/private_python.py','-c','import sys; print(sys.executable); print(sys.version)'])
print(json.dumps(r,indent=2))
