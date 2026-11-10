"""Current-host eager reference and fullgraph compiler opponents, same device timer."""
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(HERE.with_name('arithmetic9_20261027'))]

def make(case, library):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    import benchmarks.baselines as definitions
    fn = case.reference
    if library == 'torch-compile':
        import torch
        import torch_npu
        from forbid_events import install
        install(torch, torch_npu)
        from compile_msprof import install as install_compiler_timer
        install_compiler_timer(torch, torch_npu, HERE / 'baseline/torch-compile-autotune')
        fn = definitions.compiled_reference(fn, preserve_precision=True)
    files = [Path(__file__), HERE / 'baselines.py', Path(definitions.__file__),
             ROOT / 'TileOPs/workloads/elementwise.py']
    if library == 'torch-compile':
        files.extend([HERE / 'forbid_events.py', HERE / 'compile_msprof.py'])
    return Candidate(Implementation(fn), library + '::manifest_reference',
                     'Original manifest reference on NPU; compiled variant uses the existing fullgraph=True factory. Independent CPU oracle before timing; no Event fallback.',
                     {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})

from benchmarks.ascend_all_pairs import registry
prior = registry.candidate_factory
registry.candidate_factory = lambda case, library: (lambda: make(case, library)) if library in ('torch', 'torch-compile') else prior(case, library)
runpy.run_path(str(HERE / 'baselines.py'), run_name='__main__')
