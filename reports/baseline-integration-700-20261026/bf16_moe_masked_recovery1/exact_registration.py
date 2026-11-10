"""Unpublished exact-key factory, prepared for parent registration review."""
from pathlib import Path
import hashlib
import importlib.util
KEY=('ops-transformer','MoEGroupedGemmFwdOp','ds-v3-ep8-masked-bfloat16')
ROOT=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_moe_masked_recovery1')
def make_candidate(case):
 from benchmarks.ascend_baselines import UnsupportedCase
 if (KEY[0],case._call.signature.name,case.id)!=KEY:
  raise UnsupportedCase('Outside reviewed masked MoE exact key')
 pins=__import__('json').loads((ROOT/'FROZEN.json').read_text())['sha256']
 paths=[ROOT/'adapter.py',ROOT.parent/'bf16_moe_hadamard_parent10/adapter.py',ROOT.parent/'bf16_local_gemm18/binding.py']
 for p in paths:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=pins[str(p)]:raise RuntimeError('Frozen adapter changed: '+str(p))
 spec=importlib.util.spec_from_file_location('masked_recovery1_verified_adapter',paths[0]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.make_candidate(case)
