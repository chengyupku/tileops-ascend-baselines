"""Exact, CPU-safe metadata admission for this isolated 105-case proposal."""
import json
from pathlib import Path

ADMISSION_PATH = Path(__file__).with_name('omission_admission.json')
ADMISSIONS = json.loads(ADMISSION_PATH.read_text())['entries']
NEW_OPS = frozenset(e['op'] for e in ADMISSIONS if not e['op'].startswith('Conv'))

def _key(op, p):
    return op, json.dumps(p, sort_keys=True, separators=(',', ':'))

_INDEX = {_key(e['op'], e['params']): e for e in ADMISSIONS}
assert len(_INDEX) == len(ADMISSIONS)

def omission_decision(op, p):
    item = _INDEX.get(_key(op, p))
    if item is None:
        return None
    suffix = (' Original sparse-attention extension/OPP deployment and hardware MAX boundary '
              'semantics remain unverified.' if item['family'] == 'extrema' else
              ' Original Triton device compilation/execution remains unverified.')
    return item['status'], ('Exact scoped candidate ' + item['finding_id'] + '/' + item['case_id'] +
        ': complete outputs, per-call packing and whole reduction domains; CPU simulation is not native evidence.' + suffix)
