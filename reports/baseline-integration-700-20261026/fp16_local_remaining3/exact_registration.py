"""Unpublished exact-key factory for parent review; never writes shared registry."""
from pathlib import Path
import json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parent
def make_candidate(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    key=('ops-transformer',case._call.signature.name,case.id)
    expected=next((r['manifest_params'] for r in json.loads((ROOT/'CASES.json').read_text()) if tuple(r['key'])==key),None)
    if expected is None or any(case._call.indices.get(k)!=v for k,v in expected.items()):
        raise UnsupportedCase('Exact FP16 parameters do not match reviewed manifest')
    result=json.loads((ROOT/'RESULT.json').read_text())
    if not any(tuple(r['key'])==key and r['status']=='verified_pending_parent_review' for r in result['rows']):
        raise UnsupportedCase('Exact FP16 key lacks full original checker evidence')
    pins=json.loads((ROOT/'FROZEN.json').read_text())['sha256']
    for p in [ROOT/'adapter.py',ROOT.parent/'bf16_local_gemm18/binding.py']:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=pins[str(p)]:raise RuntimeError('Frozen source changed: '+str(p))
    spec=importlib.util.spec_from_file_location('fp16_local_remaining3_adapter',ROOT/'adapter.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.make_candidate(case)
