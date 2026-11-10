python3 - <<'PY'
from pathlib import Path
import shutil,json,hashlib
r=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');(r/'checkpoint_v2_bin').mkdir()
for n in ['make','gmake']:
 p=r/'checkpoint_v2_bin'/n;shutil.copy2(r/'checkpoint_make_v2.py',p);p.chmod(0o755)
s=(r/'RESUME_TARGET1_CHECKPOINT.sh').read_text().replace('compile1_checkpoint.','compile0.').replace('flash_attn_ascend950_1','flash_attn_ascend950_0').replace('$R/checkpoint_bin','$R/checkpoint_v2_bin')
(r/'RESUME_TARGET0.sh').write_text(s)
b=r.parent/'attention_native_build5/attempt03/stage/source/build'
arts=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size} for p in (b/'binary/ascend950/bin/flash_attn').glob('*') if p.is_file()]
(r/'BF16_ARTIFACTS.json').write_text(json.dumps(arts,indent=2));print(arts)
PY
