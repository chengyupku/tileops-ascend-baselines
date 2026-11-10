"""Only invoked by launch.py inside the live foreground reservation."""
import sys,runpy,platform
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
import registration as M
M.require_guard()
cid,output=sys.argv[1:];pin=M.metadata()['RUNTIME_PINS.json'][cid];M.environment(pin)
W=Path('/root/dyq/workspace' if platform.machine()=='x86_64' else '/opt/tileops-all-pairs/workspace')
S=W/'reports/baseline-integration-700-20261026/threehost_linear49/snapshot'
sys.path[:0]=[str(S/'TileOPs'),str(S/'TileOPs/src'),str(S)]
if platform.machine()=='aarch64':sys.path.extend(['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_vllm/python-deps'),str(W/'.cache/all-pairs/remote_native_tla/python-deps')])
checker=S/'TileOPs/scripts/ascend_pair_check.py'
# Original checker is executed at its original path, retaining ROOT semantics.
import hashlib
if hashlib.sha256(checker.read_bytes()).hexdigest()!=M.metadata()['LOCAL_PINS.json']['closure/evidence/ascend_pair_check.py']:raise RuntimeError('Original checker changed')
import setuptools
import benchmarks.ascend_all_pairs.registry as registry
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
private=dict(PROVIDERS);M.install(private);previous=registry.candidate_factory
registry.candidate_factory=lambda case,library:(lambda:private['DeltaNetFwdOp'](case)) if library=='ops-transformer' and case._call.signature.name=='DeltaNetFwdOp' else previous(case,library)
sys.argv=[str(checker),'--op','DeltaNetFwdOp','--case-id',cid,'--library','ops-transformer','--output',output]
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 M.require_guard()
 import torch
 if torch.npu.is_initialized():torch.npu.synchronize()
