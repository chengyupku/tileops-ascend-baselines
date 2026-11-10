"""CPU-only stdlib manifest preparation; no baseline/runtime imports."""
import json,sys,hashlib
from pathlib import Path
from baseline_cache_guard_binding import capture_guard
out=Path(sys.argv[1]); root=Path(__file__).resolve().parents[1]
t=json.loads((out/'task.json').read_text())
assert t['target_aicore_count']==32 and t['host']=='server1'
files=['tools/measure_baseline_retest32_server1.py','TileOPs/scripts/cache_frozen_pair_check.py','tools/measure_baselines.py']
proof={'cpu_only':True,'kind':'manifest and original checker/source fingerprint preparation; native prebuilt reuse','sources':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},'guard_before':capture_guard()}
(out/'prepare.json').write_text(json.dumps(proof))
