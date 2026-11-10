from pathlib import Path
import sys,json,platform,hashlib,os,shutil
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');V=W/'ascend_baselines/tilelang-ascend'
files=[V/'examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py',V/'examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py',V/'src/tl_templates/ascend/common.h',P/'frozen/TileOPs/scripts/ascend_pair_check.py']+list((W/'.cache/all-pairs/remote_tla_compiler/handoff/lib').glob('*.so'))
r=dict(machine=platform.machine(),python=sys.version,executable=sys.executable,bisheng=shutil.which('bisheng'),sdk=os.environ.get('ASCEND_HOME_PATH'),files=[])
for f in files:
 if not f.is_file():r['files'].append(dict(path=str(f),exists=False));continue
 b=f.read_bytes();r['files'].append(dict(path=str(f),exists=True,sha256=hashlib.sha256(b).hexdigest(),elf_machine=int.from_bytes(b[18:20],'little') if b[:4]==b'\x7fELF' else None))
(P/'PREFLIGHT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
