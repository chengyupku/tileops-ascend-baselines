"""CPU-only, exact manifest contract gate for this reviewed omission proposal.

Wider inputs must be reviewed explicitly; this does not generalize a static audit
into an unsupported claim of universal native correctness.
"""
import json
from pathlib import Path
from functools import lru_cache

@lru_cache(None)
def records():
    return json.loads(Path(__file__).with_name('omission_cases.json').read_text())

def signature(params):
    if params.get('ord') == float('inf'):
        params = dict(params, ord='inf')
    return json.dumps(params,sort_keys=True,separators=(',',':'))

@lru_cache(None)
def contracts():
    return {(r['op'],signature(r['params'])) for r in records()}

OPS = frozenset(r['op'] for r in records())

def eligible(op,params):
    return (op,signature(params)) in contracts()
