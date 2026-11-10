"""Original TileLang public APIs under the unchanged full checker."""
import ast
import dataclasses
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.tilelang import adapters, attention_crops
from benchmarks.ascend_baselines import source_hashes
path=Path(adapters.__file__)
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sum_mean')
assert isinstance(node.body[0],ast.Raise) and 'constant-projection GEMM reduction is retired' in ast.unparse(node.body[0])
# The user restored complete public API compositions on 2026-10-27.
# This is a complete FP32 GEMM, not the explicitly retired test-kernel reduction.
node.body=node.body[1:]
namespace=dict(adapters.__dict__)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
def factory(case,library):
    assert library=='tilelang'
    def build():
        provider={'SumFwdOp':namespace['sum_mean'],'GQADenseFwdOp':attention_crops.dense,'GQAPagedFwdOp':attention_crops.paged}[case._call.signature.name]
        c=provider(case)
        return dataclasses.replace(c,source_sha256=c.source_sha256|source_hashes(Path(__file__)))
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
