from pathlib import Path
import json,hashlib,platform,sys,subprocess,datetime
W=Path('/opt/tileops-all-pairs/workspace');C=W/'.cache/all-pairs/remote_tla_compiler';V=W/'ascend_baselines/tilelang-ascend';P=W/'reports/baseline-integration-700-20261026/surge_tla_attention97/worker06'
paths=[C/'handoff/lib/libtilelang.so',C/'handoff/lib/libtvm.so',V/'tilelang/jit/adapter/cython/cython_wrapper.pyx',V/'3rdparty/pto-isa/include/pto/pto-inst.hpp',Path('/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/bisheng')]
r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),machine=platform.machine(),assets=[])
for p in paths:
 d=p.read_bytes();r['assets'].append(dict(path=str(p),sha256=hashlib.sha256(d).hexdigest(),elf_machine=int.from_bytes(d[18:20],'little') if d[:4]==b'\x7fELF' else None))
r['python_abi']=subprocess.run(['/opt/tileops-ci/runtime/bin/python','-I','-S','-B','-c','import sys,sysconfig;print(sys.version);print(sysconfig.get_config_var("SOABI"))'],capture_output=True,text=True).stdout
r['compiler_version']=subprocess.run([str(paths[-1]),'--version'],capture_output=True,text=True).stdout
for p in paths[:2]:
 s=subprocess.run(['nm','-D',str(p)],capture_output=True,text=True);r[str(p)+'_exports']=[l for l in s.stdout.splitlines() if 'TVMFuncGetGlobal' in l or 'TileLang' in l][:12]
print(json.dumps(r,indent=2))
