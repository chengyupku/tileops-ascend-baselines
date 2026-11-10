import subprocess,pathlib,json,platform,hashlib
W=pathlib.Path('/root/dyq/workspace') if platform.machine()=='x86_64' else pathlib.Path('/opt/tileops-all-pairs/workspace')
def cmd(c):
 r=subprocess.run(c,capture_output=True,text=True,timeout=30);return dict(cmd=c,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
r=dict(hardware=cmd(['npu-smi','info']),headers=[],original_sources=[])
root=W/'.cache/all-pairs/remote_native_deepgemm_deepep/official-asc-devkit'
if root.exists():r['official_headers']=[cmd(['git','-C',str(root),'rev-parse','HEAD']),cmd(['git','-C',str(root),'diff','--name-only'])]
for f in [pathlib.Path('/usr/local/Ascend/cann/asc/include/c_api/sys_var/sys_var.h'),pathlib.Path('/usr/local/Ascend/cann/asc/include/c_api/cube_datamove/cube_datamove.h')]:
 if f.exists():r['headers'].append(dict(path=str(f.resolve()),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
for rel in ['examples/gemm/example_gemm.py','examples/batch_gemm/batch_gemm.py','examples/causal_conv1d/causal_conv1d_decode.py','examples/causal_conv1d/causal_conv1d_pto.py','examples/deepseek_v4/int8_gemm.py']:
 f=W/'ascend_baselines/tilelang-ascend'/rel
 if f.exists():r['original_sources'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),decorators=[l for l in f.read_text().splitlines() if 'target=' in l or '@tilelang.jit' in l]))
print(json.dumps(r,indent=2))
