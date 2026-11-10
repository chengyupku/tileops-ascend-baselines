"""Read-only source/library identity capture; no device runtime imports."""
import pathlib,hashlib,json,platform,os
P=pathlib.Path(__file__).resolve().parent;S=pathlib.Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
paths=[S/'torch_npu/lib/libtorch_npu.so',S/'torch_npu/lib/libnpu_profiler.so']
paths += list((S/'torch_npu/_inductor').rglob('*.py'))
paths += list((S/'torch_npu/dynamo/torchair/configs').rglob('*.py'))
paths += list((S/'torch_npu/csrc/aten').glob('*yaml'))
paths += [pathlib.Path('/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/bisheng'),pathlib.Path('/usr/local/Ascend/cann-9.1.0/version.cfg')]
obj={'machine':platform.machine(),'host':platform.node(),'sources':{str(p):sha(p) for p in paths if p.is_file()},'missing':[str(p) for p in paths if not p.exists()],'memory_current':pathlib.Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text().strip(),'memory_limit':pathlib.Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text().strip(),'device_imports':False}
(P/'STATIC_ENVIRONMENT.json').write_text(json.dumps(obj,indent=2))
print('STATIC_ENVIRONMENT',len(obj['sources']))
