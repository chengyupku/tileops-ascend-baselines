from pathlib import Path
import sys,platform,json,hashlib,subprocess,shutil
W=Path(sys.argv[1]);local=str(W).startswith('/root');V=W/'ascend_baselines/tilelang-ascend'
C=W/'.cache/all-pairs/remote_tla_compiler/handoff' if not local else W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build/source/build'
libs=list((C/'lib').glob('*.so')) if not local else list(C.glob('libtilelang*.so'))+list((C/'tvm').glob('libtvm*.so'))
paths=[V/'examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py',V/'examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py',V/'examples/flash_attention/flash_attn_bhsd.py',V/'src/tl_templates/pto/common.h',V/'3rdparty/pto-isa/include/pto/pto-inst.hpp',V/'tilelang/jit/adapter/cython/adapter.py']+libs
assets=[]
for p in paths:
 if not p.is_file():assets.append(dict(path=str(p),exists=False));continue
 b=p.read_bytes();assets.append(dict(path=str(p),exists=True,sha256=hashlib.sha256(b).hexdigest(),elf_machine=int.from_bytes(b[18:20],'little') if b[:4]==b'\x7fELF' else None))
python= W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build/venv/bin/python' if local else Path('/opt/tileops-ci/runtime/bin/python')
r=subprocess.run([str(python),'-I','-S','-c','import sys,sysconfig;print(sys.version);print(sysconfig.get_config_var("SOABI"))'],capture_output=True,text=True)
print(json.dumps(dict(machine=platform.machine(),python=str(python),abi=r.stdout,abi_rc=r.returncode,assets=assets,library_count=len(libs),guard=shutil.which('npu-guard')),indent=2))
