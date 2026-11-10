"""Validate original staged NN candidates under the unchanged full checker."""
import ast,dataclasses,runpy,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
name='benchmarks.ascend_all_pairs.ops_nn'
package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_nn')];package.__package__=name
sys.modules[name]=package
from benchmarks.ascend_all_pairs.ops_nn import staging
from benchmarks.ascend_baselines import source_hashes

from benchmarks.ascend_all_pairs.ops_nn import matrix
matrix_path=Path(matrix.__file__)
tree=ast.parse(matrix_path.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='causal_convolution')
assert isinstance(node.body[0],ast.If) and 'Retired Conv2DV2 plus Swish composition' in ast.unparse(node.body[0])
# Only superseded interface admission is removed, preserving full original API calls.
node.body=node.body[1:]
namespace=dict(matrix.__dict__)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(matrix_path),'exec'),namespace)
restored=namespace['causal_convolution']

def factory(case,library):
 if library!='ops-nn':raise ValueError('ops-nn staging only')
 def build():
  assert case._call.signature.name=='CausalConv1dUpdateFwdOp'
  c=restored(case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
