"""Record original TLA launches, including adapters restored from disk cache."""
from functools import wraps
import inspect
from pathlib import Path

from benchmarks.ascend_all_pairs.jit_evidence import Evidence


def tilelang_run(run, source_hashes, cache):
    evidence = Evidence(source_hashes, cache, 'tilelang')

    @wraps(run)
    def wrapped(*args, **kwargs):
        from . import native
        native.runtime()
        records = list(native._deployment['selected_libraries'].values())
        if native._deployment['cython_cache']['exists']:
            records.append(native._deployment['cython_cache'])
        for record in records:
            if evidence.file(record['file'])['sha256'] != record['sha256']:
                raise RuntimeError('Compiler deployment artifact changed before launch')
        from tilelang.jit.kernel import JITKernel
        original = JITKernel.__call__
        evidence.file(inspect.getsourcefile(JITKernel))
        observed = []

        def traced(kernel, *values, **options):
            adapter = kernel.adapter
            loaded = getattr(getattr(adapter, 'lib', None), '_name', None)
            if not isinstance(loaded, (str, bytes)):
                raise RuntimeError('TileLang adapter has no loaded CDLL identity')
            path = Path(loaded.decode() if isinstance(loaded, bytes) else loaded).resolve()
            declared = adapter.libpath
            if declared is not None and Path(declared).resolve() != path:
                raise RuntimeError('TileLang declared path disagrees with loaded CDLL')
            # Original CythonKernelAdapter.from_database loads kernel_lib_path
            # but leaves lib_generator.libpath=None. The CDLL handle still
            # identifies the real file; do not mutate the vendor adapter.
            binary = evidence.file(path)
            source = evidence.blob(kernel.get_kernel_source(), '.kernel.txt')
            adapter_source = evidence.file(inspect.getsourcefile(type(adapter)))
            result = original(kernel, *values, **options)
            if evidence.file(path) != binary:
                raise RuntimeError('TileLang binary changed during launch')
            observed.append({'binary':binary, 'generated_source':source,
                             'adapter_source':adapter_source,
                             'declared_libpath':declared, 'loaded_cdll':str(path),
                             'target':str(kernel.target),
                             'execution_backend':kernel.execution_backend})
            return result

        JITKernel.__call__ = traced
        try:
            result = run(*args, **kwargs)
            if not observed:
                raise RuntimeError('No original TileLang JITKernel launch observed')
            evidence.calls.extend(observed)
            evidence.flush()
            return result
        finally:
            JITKernel.__call__ = original
    return wrapped
