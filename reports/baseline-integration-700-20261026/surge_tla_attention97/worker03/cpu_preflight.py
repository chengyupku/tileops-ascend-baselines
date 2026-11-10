from pathlib import Path
import json,hashlib,platform,sys,subprocess
P=Path(__file__).resolve().parent
W=Path('/opt/tileops-all-pairs/workspace') if platform.machine()=='aarch64' else Path('/root/dyq/workspace')
C=W/'.cache/all-pairs/remote_tla_compiler/handoff/lib' if platform.machine()=='aarch64' else W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build/source/build'
r={'machine':platform.machine(),'cpu_inspector_python':sys.version,'source_identity':[],'libraries':[],'templates':[]}
for f in (P/'original_sources').rglob('*.py'):
 original=W/'ascend_baselines/tilelang-ascend'/f.relative_to(P/'original_sources');h=hashlib.sha256(f.read_bytes()).hexdigest();r['source_identity'].append({'private':str(f),'original':str(original),'sha256':h,'matches':original.exists() and hashlib.sha256(original.read_bytes()).hexdigest()==h})
for f in C.glob('*.so'):
 with f.open('rb') as stream:b=stream.read(20)
 r['libraries'].append({'path':str(f),'elf_machine':int.from_bytes(b[18:20],'little'),'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns})
for x in ['src/tl_templates/pto/common.h','3rdparty/pto-isa/include/pto/pto-inst.hpp','src/tl_templates/ascend/common.h']:
 f=W/'ascend_baselines/tilelang-ascend'/x;r['templates'].append({'path':str(f),'exists':f.exists()})
(P/'CPU_PREFLIGHT_HOST.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
