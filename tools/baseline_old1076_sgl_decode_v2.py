"""Select the existing complete decode API for independent query rows."""
import ast
import dataclasses
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.sgl import attention
from benchmarks.ascend_baselines import source_hashes
path=Path(attention.__file__)
tree=ast.parse(path.read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build')
branches=[n for n in ast.walk(node) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('1 < nq <= nk')]
assert len(branches)==1
# Choose the already existing alternate original API; neither kernel changes.
branches[0].test=ast.Constant(False)
namespace=dict(attention.__dict__)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
def factory(case,library):
    assert library=='sgl-kernel-npu'
    def build():
        op=case._call.signature.name
        assert op in ('GQAVarlenFwdOp','GQAPagedFwdOp')
        c=namespace['_build'](case,paged=op=='GQAPagedFwdOp')
        return dataclasses.replace(c,source_sha256=c.source_sha256|source_hashes(Path(__file__)),adaptation=c.adaptation+' Select original decode API for each independent query using existing exact context cropping; no partial-score reduction, kernel, compiler or schedule change.')
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
