# stdlib-only preflight: no torch, torch_npu or ACL imports.
from pathlib import Path
import hashlib,json,ast,sys
P=Path(__file__).resolve().parent; W=Path('/opt/tileops-all-pairs/workspace')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=json.loads((P/'PREFLIGHT_LOCAL.json').read_text())['original_hashes']
paths=[W/k for k in expected]
paths += list((W/'TileOPs/benchmarks/ascend_all_pairs/sgl').rglob('*.py'))
paths += [W/'TileOPs/benchmarks/ascend_all_pairs/registry.py',W/'TileOPs/benchmarks/ascend_baselines.py',W/'TileOPs/src/tileops/manifest/spec/linear_attention.yaml',W/'tools/baseline_protocol.py']
paths += list(Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu/fla').glob('*.py'))
state={str(p):h(p) for p in paths}
phase=sys.argv[1]
(P/('SHARED_HASHES_'+phase+'.json')).write_text(json.dumps(state,indent=2))
if phase=='before':
 assert all(h(W/k)==v for k,v in expected.items())
 for p in P.glob('*.py'):ast.parse(p.read_text())
 deps=['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/recovery_1622/tasks/fla-bf16/runtime/deps')]
 (P/'REMOTE_PREFLIGHT.json').write_text(json.dumps(dict(source_hashes_match=True,imports='stdlib only; device imports forbidden',dependencies={p:Path(p).is_dir() for p in deps}),indent=2))
 assert all(Path(p).is_dir() for p in deps)
else:
 before=json.loads((P/'SHARED_HASHES_before.json').read_text()); assert state==before
 (P/'SHARED_UNCHANGED.json').write_text(json.dumps(dict(unchanged=True,files=len(state))))
