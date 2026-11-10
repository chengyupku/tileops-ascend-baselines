import sys,os,json,importlib.util,runpy
from pathlib import Path
ROOT=Path('/opt/tileops-baseline-cache/workspace');HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT),str(ROOT/'tools')]
task=json.loads(Path(sys.argv[sys.argv.index('--task')+1]).read_text());lib=task['key'][0];kind={'tilelang-ascend':'tla','vllm-ascend':'vllm','catlass':'catlass','ops-tensor':'ops_tensor'}[lib]
import benchmarks.ascend_all_pairs
if kind=='tla' and task['key'][1]=='BmmFwdOp':
 import hashlib
 regpath=ROOT/'reports/baseline-integration-700-20261026/threehost_pto_registration23/registration_parent.py'
 assert hashlib.sha256(regpath.read_bytes()).hexdigest()=='c3a29a8c68172513f1d5a63dab4ec135111a3f7f14d87a93edfc5ff2a2e6e73b'
 import types,dataclasses,tempfile,shutil
 owner='v3-full/server1/'+task['id'];os.environ['PTO23_OWNER']=owner
 reg=types.ModuleType('_v3_original_pto23');reg.__file__=str(regpath);sys.modules[reg.__name__]=reg
 # Only adapt historical campaign owner literal; all lock/PID/ABI/compiler checks remain.
 original=regpath.read_text()
 assert original.count('baseline700-20261026/threehost_pto_registration23')==1
 exec(compile(original.replace('baseline700-20261026/threehost_pto_registration23',owner),str(regpath),'exec'),reg.__dict__)
 out=Path(sys.argv[sys.argv.index('--out')+1]);runs=regpath.parent/'runs';runs.mkdir(exist_ok=True)
 private_root=Path(tempfile.mkdtemp(prefix='v3full-',dir=runs));os.environ['PTO23_RUN_ROOT']=str(private_root)
 from benchmarks.ascend_all_pairs import registry
 from benchmarks.ascend_baselines import source_hashes
 def original_bmm(case,library):
  assert library=='tilelang-ascend' and reg.matches(case), 'Original PTO BMM contract required'
  def build():
   candidate=reg.make(case);call=candidate.implementation.run;stable=[]
   def run(*args,**kwargs):
    prior=os.environ.get('ASCEND_WORK_PATH')
    if stable:
     if stable[0] is None:os.environ.pop('ASCEND_WORK_PATH',None)
     else:os.environ['ASCEND_WORK_PATH']=stable[0]
    try:
     result=call(*args,**kwargs)
     if not stable:stable.append(os.environ.get('ASCEND_WORK_PATH'))
     return result
    finally:
     # Profiling session is already initialized; restore its output path after native call.
     if prior is None:os.environ.pop('ASCEND_WORK_PATH',None)
     else:os.environ['ASCEND_WORK_PATH']=prior
   def cleanup():
    if stable:
     if stable[0] is None:os.environ.pop('ASCEND_WORK_PATH',None)
     else:os.environ['ASCEND_WORK_PATH']=stable[0]
    if candidate.cleanup:candidate.cleanup()
   return dataclasses.replace(candidate,implementation=dataclasses.replace(candidate.implementation,run=run),cleanup=cleanup,source_sha256={**candidate.source_sha256,**source_hashes(Path(__file__))})
  return build
 registry.candidate_factory=original_bmm
 try:runpy.run_path(str(HERE/'bootstrap.py'),run_name='__main__')
 finally:
  os.chdir(ROOT)
  shutil.move(str(private_root),str(out/'pto_cache'))
 raise SystemExit(0)
name='benchmarks.ascend_all_pairs.'+kind
spec=importlib.util.spec_from_file_location(name,HERE/'providers'/kind/'__init__.py',submodule_search_locations=[str(HERE/'providers'/kind)])
m=importlib.util.module_from_spec(spec);sys.modules[name]=m;setattr(benchmarks.ascend_all_pairs,kind,m);
if kind=='tla':
 import importlib,functools
 native=importlib.import_module(name+'.native')
 adapters=importlib.import_module(name+'.adapters')
 def make(op,case):
  native.prepare_deployment();return adapters.factory(op,case)
 m.PROVIDERS={op:functools.partial(make,op) for op in importlib.import_module(name+'.plans').IMPLEMENTED}
else:spec.loader.exec_module(m)
if kind in ('catlass','ops_tensor'):
 import importlib
 importlib.import_module(name+'.build')
for n,mod in list(sys.modules.items()):
 if n.startswith(name+'.'):
  if hasattr(mod,'WORKSPACE'):mod.WORKSPACE=ROOT
  if hasattr(mod,'VENDOR'):mod.VENDOR=ROOT/'ascend_baselines'/lib
  if hasattr(mod,'CACHE'):mod.CACHE=ROOT/'.cache/all-pairs'/kind
if kind=='tla':
 sys.path.insert(0,str(ROOT/'.cache/all-pairs/remote_tla_compiler/python'));sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
 import setuptools
 from benchmarks.ascend_all_pairs.tla.native import runtime
 tl=runtime();out=Path(sys.argv[sys.argv.index('--out')+1]);cache=out/'jit_kernel_cache';cache.mkdir();tl.set_cache_dir(str(cache))
if kind=='tla' and task['key'][1] not in m.PROVIDERS:
 import ast,hashlib,types
 source=ROOT/'tools/baseline_support_tla_composed_v3.py'
 expected='4a93a291f880156d5af1b6574bf06a7827626bc85a3155025891595c0674a0ae'
 if task['key'][1] in ('ProdFwdOp','CumprodFwdOp'):
  source=ROOT/'tools/baseline_support_tla_composed_v4.py'
  expected='0acb4d32c1b4ef29662018f30d99d92609d3868ea61a93965380aa87b5dddee2'
 if task['key'][1] in ('LogSumExpFwdOp','LogSoftmaxFwdOp'):
  source=ROOT/'tools/baseline_old1076_tla_block_v3.py'
  expected='6e205dada0eb942219d8012e78199417e614a23cdcf4b2c885848fa152bfae63'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
 tree=ast.parse(source.read_text(),filename=str(source))
 assert ast.unparse(tree.body[-2])=='registry.candidate_factory = factory'
 assert isinstance(tree.body[-1],ast.Expr) and 'runpy.run_path' in ast.unparse(tree.body[-1])
 tree.body=tree.body[:-2]  # load original definitions without its old standalone launcher
 comp=types.ModuleType('_v3_original_composition');comp.__file__=str(source)
 sys.modules[comp.__name__]=comp;exec(compile(tree,str(source),'exec'),comp.__dict__)
 if task['key'][1] in comp.OPS:
  m.PROVIDERS[task['key'][1]]=lambda case:comp.factory(case,'tilelang-ascend')()
runpy.run_path(str(HERE/'bootstrap.py'),run_name='__main__')
