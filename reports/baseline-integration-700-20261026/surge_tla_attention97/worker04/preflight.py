from pathlib import Path
import json,platform,sys,sysconfig,hashlib,os,shutil
W=Path('/opt/tileops-all-pairs/workspace');P=Path(__file__).resolve().parent
paths=[W/'ascend_baselines/tilelang-ascend/tilelang/__init__.py',W/'ascend_baselines/tilelang-ascend/examples/gqa_fwd_varlen/gqa_fwd_varlen.py',W/'.cache/all-pairs/remote_tla_compiler/handoff/lib/libtilelang.so',W/'.cache/all-pairs/remote_tla_compiler/handoff/lib/libtvm.so',W/'ascend_baselines/tilelang-ascend/3rdparty/pto-isa/include/pto/pto-inst.hpp',P/'frozen/TileOPs/scripts/ascend_pair_check.py',P/'private_tla/native.py']
r=dict(host=platform.node(),arch=platform.machine(),python=sys.version,executable=sys.executable,abi=sysconfig.get_config_var('SOABI'),cwd=os.getcwd(),files=[])
for p in paths:
 d=dict(path=str(p),exists=p.is_file())
 if p.is_file():
  with p.open('rb') as f:h=f.read(20)
  d.update(sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),elf_machine=int.from_bytes(h[18:20],'little') if h[:4]==b'\x7fELF' else None)
 r['files'].append(d)
print(json.dumps(r,indent=2))
