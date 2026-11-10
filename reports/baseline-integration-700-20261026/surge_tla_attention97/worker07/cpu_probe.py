from pathlib import Path
import platform,sys,json,hashlib,subprocess,os
remote=platform.machine()=='aarch64';W=Path('/opt/tileops-all-pairs/workspace') if remote else Path('/root/dyq/workspace');V=W/'ascend_baselines/tilelang-ascend'
C=W/'.cache/all-pairs/remote_tla_compiler' if remote else W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build'
paths=[V/'examples/gqa_fwd_varlen/gqa_fwd_varlen.py',V/'tilelang/__init__.py',V/'3rdparty/pto-isa/include/pto/pto-inst.hpp',V/'tilelang/jit/adapter/cython/adapter.py',W/'TileOPs/scripts/ascend_pair_check.py']
paths+=list((C/'handoff/lib' if remote else C/'source/build').glob('*.so'))
if not remote: paths+=list((C/'source/build/tvm').glob('*.so'))
r={'host':platform.node(),'arch':platform.machine(),'python':sys.version,'executable':sys.executable,'cwd':os.getcwd(),'assets':[]}
for p in paths:
 if not p.is_file():r['assets'].append({'path':str(p),'missing':True});continue
 b=p.read_bytes();r['assets'].append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'elf_machine':int.from_bytes(b[18:20],'little') if b[:4]==b'\x7fELF' else None})
print(json.dumps(r,indent=2))
