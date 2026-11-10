"""Private process-only provider; original full checker is executed unchanged."""
import os, sys, pathlib, tempfile, json, hashlib, runpy
R=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/opt/tileops-all-pairs/workspace')
os.chdir(R)
(R/'tmp').mkdir(exist_ok=True)
tempfile.tempdir=f'/proc/{os.getpid()}/cwd/tmp'
os.environ['TMPDIR']=tempfile.tempdir
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT),str(R)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import Candidate
from benchmarks.api import Implementation
from firstcase_adapter import firstcase
OLD=registry.candidate_factory
SOURCE=R.parent/'attention_native_build5/attempt03/stage/source'
PINS={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'firstcase_adapter.py',R/'firstcase_runner.py',SOURCE/'attention/flash_attn/op_kernel/flash_attn.cpp',SOURCE/'attention/flash_attn/torch_extension/flash_attn.py'] if p.is_file()}
launch=[]
def run(*args):
    result=firstcase(*args)
    import torch
    torch.npu.synchronize()
    launch.append({'api':'original flash_attn_metadata + flash_attn','returned_and_synchronized':True})
    return result
def factory(case,library):
    assert library=='ops-transformer' and case.id=='gpt-oss-sink-decode-bfloat16-bfloat16'
    assert case._call.signature.name=='GQAPagedFwdOp'
    return lambda:Candidate(Implementation(run),'cann_ops_transformer.flash_attn_metadata + flash_attn','Original API, one last-position token per request; mode0 sees all cached keys; unchanged sink',dict(PINS))
registry.candidate_factory=factory
try:
    runpy.run_path(str(ROOT/'TileOPs/scripts/ascend_pair_check.py'),run_name='__main__')
finally:
    maps=pathlib.Path('/proc/self/maps').read_text();(R/'firstcase.maps').write_text(maps)
    paths=set(line.split()[-1] for line in maps.splitlines() if '/' in line and any(s in line for s in ['.so','kernel','cann_ops']))
    pins={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sorted(paths) if pathlib.Path(p).is_file()}
    (R/'firstcase.native.json').write_text(json.dumps({'launch':launch,'loaded_files_sha256':pins,'exact_kernel_binary':'not yet attributed; inspect runtime logs and binary metadata','private_registry':True},indent=2))
