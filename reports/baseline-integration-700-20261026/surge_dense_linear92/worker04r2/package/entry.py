import sys,os,json,runpy
from pathlib import Path
from direct_acl import require_guard
P=Path(__file__).resolve().parent;W=P.parent;R=Path("/opt/tileops-all-pairs/workspace")
require_guard();r=json.loads((P/'RUNTIME_RECIPE.json').read_text());sys.path[:0]=[str(R/'TileOPs'),str(R/'TileOPs/src'),str(R)];sys.path.extend(r['python_paths'])
import setuptools
import benchmarks.ascend_all_pairs.registry as registry
from binding import build
import capture_hooks
capture_hooks.install()
original=registry.candidate_factory
registry.candidate_factory=lambda case,lib: (lambda:build(case)) if lib=='ops-transformer' and case._call.signature.name=='GDNFwdOp' else original(case,lib)
runpy.run_path(r['checker'],run_name='__main__')
