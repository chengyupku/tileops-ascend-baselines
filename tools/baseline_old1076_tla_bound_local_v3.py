"""Original local PTO APIs with restored reviewed top-level compositions."""
import dataclasses
import importlib
import runpy
import sys
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.tla import native,plans,adapters
from benchmarks.ascend_baselines import source_hashes
package=types.ModuleType('old1076_tla_interfaces')
package.__path__=[str(ROOT/'reports/baseline-old1076-20261029/tla_interfaces')]
sys.modules[package.__name__]=package
sys.modules[package.__name__+'.native']=native
sys.modules[package.__name__+'.plans']=plans
omissions=importlib.import_module(package.__name__+'.omissions')
original_load=native.load
def load(relative,*names):
    module=original_load(relative,*names)
    if relative=='examples/cann-bench/group_norm/example_group_norm.py':
        original_namespace=types.SimpleNamespace(**module.__dict__)
        for name in ('group_norm_kernel_serial','group_norm_kernel_cpipeline'):
            if hasattr(module,name):
                def select(*args,_name=name,**kwargs):
                    return native.compile_builder(original_namespace,_name,*args,**kwargs)
                setattr(module,name,select)
    return module
native.load=load
def factory(case,library):
    assert library=='tilelang-ascend'
    def build():
        op=case._call.signature.name
        if omissions.eligible(op,case.params):
            c=omissions.make(case,op=op)
        else:
            c=adapters.factory(op,case)
        return dataclasses.replace(c,source_sha256=c.source_sha256|source_hashes(Path(__file__)),adaptation=c.adaptation+' Use original public PTO backend with original pass configuration and original host-selected tile geometry; restore owned top-level compositions without changing original kernels.')
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
