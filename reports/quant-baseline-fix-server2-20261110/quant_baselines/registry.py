"""Exact-case lookup and the existing benchmarks.api Implementation interface."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))

def available(op, case_id, params):
    """Return only measured variants for the exact declared parameters; no device imports."""
    rows = json.loads((ROOT / 'verified_cases.json').read_text())
    matched = [r for r in rows if r['op'] == op and r['case'] == case_id]
    if not matched:
        return []
    if any(_canonical(r['parameters']) != _canonical(params) for r in matched):
        raise ValueError('Case name matches but parameters differ from verified contract')
    return [r['variant'] for r in matched]

def implementations(op, case):
    """Call under host NPU guard, with the fixed original Case and locked environment."""
    from benchmarks.api import Implementation
    from . import make
    return {v: Implementation(make(op, case.params, case.reference, v))
            for v in available(op, case.id, case.params)}

def verify_reference_sources(campaign_root):
    """Verify the reference/checker/measurement source snapshot before reproducing results."""
    root = Path(campaign_root)
    for relative, expected in json.loads((ROOT / 'reference_source_hashes.json').read_text()).items():
        file = root / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original source mismatch: {relative}')
    return True
