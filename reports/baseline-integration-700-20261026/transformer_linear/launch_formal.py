"""Guarded formal-provider replay using the frozen original runtime recipe."""
from pathlib import Path
import json,sys,os,hashlib,runpy
W=Path(__file__).resolve().parent;P=W/'frozen_v1/chunk_package';R=W.parents[2]
sys.path.insert(0,str(P));from direct_acl import require_guard
require_guard();recipe=json.loads((P/'RUNTIME_RECIPE.json').read_text())
for k,v in recipe['required_environment'].items():
 if os.environ.get(k)!=v:raise RuntimeError('Runtime recipe mismatch: '+k)
for pin in json.loads((P/'RUNTIME_PINS.json').read_text())['files']:
 if hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()!=pin['sha256']:raise RuntimeError('Runtime pin changed: '+pin['path'])
sys.path[:0]=[str(R/'TileOPs'),str(R/'TileOPs/src'),str(R)];sys.path.extend(recipe['python_paths'])
import setuptools
# No factory replacement: checker -> registry -> formal provider -> frozen binding.
runpy.run_path(recipe['checker'],run_name='__main__')
