#!/usr/bin/env python3
"""Reuse the exact classic checker with an isolated provider registry."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT.parent)]
spec = importlib.util.spec_from_file_location('original_classic_checker', ROOT / 'scripts/ascend_classic_check.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.require_guard()
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from benchmarks.ops import bench_ascend_classics
bench_ascend_classics.candidate_factories = lambda case: {
    'ops-transformer': lambda: PROVIDERS[case._call.signature.name](case)
}
from benchmarks.ascend_graph_call import GraphCall
original_call = GraphCall.call
calls = []
def traced_call(self, op, inputs, outputs, attrs, **kwargs):
    result = original_call(self, op, inputs, outputs, attrs, **kwargs)
    calls.append({'op': op, 'vendor': str(self.vendor), 'api': 'aclopCompileAndExecute',
                  'inputs': [None if x is None else {'shape': list(x.shape), 'dtype': str(x.dtype)} for x in inputs],
                  'attributes': attrs, 'returned': 0})
    return result
GraphCall.call = traced_call
rc = checker.main()
out = Path(sys.argv[sys.argv.index('--output') + 1])
row = json.loads(out.read_text())
row['provider_module'] = 'benchmarks.ascend_all_pairs.ops_transformer'
row['native_graph_calls'] = calls
row['input_protection'] = 'Original workloads.numerics.Request(preserve_inputs=True)'
out.write_text(json.dumps(row, indent=2) + '\n')
raise SystemExit(rc)
