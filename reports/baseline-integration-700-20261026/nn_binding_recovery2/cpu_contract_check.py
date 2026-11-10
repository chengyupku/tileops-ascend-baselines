"""CPU-only structural stubs: no torch import, tensors, device, or native compilation."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from collections import namedtuple
ROOT = Path('/root/dyq/workspace')
source = (ROOT/'TileOPs/workloads/numerics.py').read_text()
tree = ast.parse(source)
names = {'_outputs', '_tensor_pairs', 'compare_outputs'}
# Execute only original pure control flow with metadata-only objects. Numerical
# comparison is deliberately a no-op, so these checks make no accuracy claim.
class Metadata:
    shape = (0,)
    dtype = 'stub'
    device = 'stub'
    def numel(self): return 0
class Partial: pass
class Custom: pass
class Exact:
    kind = 'exact'
    def tolerance(self, default): return default
ns = dict(Any=object, Evidence=object, torch=SimpleNamespace(Tensor=Metadata),
          Partial=Partial, Custom=Custom,
          CheckResult=namedtuple('CheckResult','checked_outputs total_outputs max_abs_err unchecked_reason'),
          reference_tolerance=lambda dtype: {}, _assert_close_in_blocks=lambda *a, **kw: None)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        exec(ast.get_source_segment(source, node), ns)
compare = ns['compare_outputs']
a = Metadata()
absent = compare((a,None,None),(a,None,None),Exact())
present = compare((a,a,a),(a,a,a),Exact())
assert (absent.checked_outputs, absent.total_outputs) == (1,3)
assert (present.checked_outputs, present.total_outputs) == (3,3)
for bad in [(a,a,a), a]:
    try: compare(bad,(a,None,None),Exact())
    except (AssertionError,ValueError): pass
    else: raise AssertionError('Changed return contract incorrectly accepted')
# Retention evidence is a source assertion, not proof of device correctness.
graph = (ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_math/graph.py').read_text()
assert "'keep':[xs,outputs]" in graph
assert graph.index('torch.npu.synchronize()', graph.index('def cleanup')) < graph.index('while pending', graph.index('def cleanup'))
print(json.dumps({'passed':True, 'checks':5, 'none_output_count':absent._asdict(),
 'tensor_output_count':present._asdict(), 'contract_mutations_rejected':2,
 'pending_retention_source_assertion':True, 'torch_imported':False,
 'numerical_validation':False, 'device_runs':0},indent=2))
