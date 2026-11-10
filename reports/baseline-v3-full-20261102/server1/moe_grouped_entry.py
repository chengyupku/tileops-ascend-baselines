"""Restore hash-pinned historical successful public graph adapter unchanged."""
import sys,ast,types,hashlib,runpy
from pathlib import Path
ROOT=Path('/opt/tileops-baseline-cache/workspace');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
p=ROOT/'tools/baseline_old1076_transformer_arm_graph_v3.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='7eb6af302c41501a5971c530ef62f617a9b16e850bff57a4e1803510572019a8'
tree=ast.parse(p.read_text(),filename=str(p))
assert isinstance(tree.body[-1],ast.Expr) and 'runpy.run_path' in ast.unparse(tree.body[-1])
tree.body=tree.body[:-1]
m=types.ModuleType('_original_moe_graph_v3');m.__file__=str(p);sys.modules[m.__name__]=m
exec(compile(tree,str(p),'exec'),m.__dict__)
runpy.run_path(str(HERE/'bootstrap.py'),run_name='__main__')
