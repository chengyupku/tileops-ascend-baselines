#!/usr/bin/env python3
"""Add existing private runtime dependencies for a guarded remote measurement."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
library, target, *args = sys.argv[1:]
if library == 'tilelang-ascend':
    sys.path[:0] = [str(ROOT / '.cache/all-pairs/remote_tla_compiler/python'),
                    str(ROOT / '.cache/all-pairs/remote_native_tla/python-deps')]
    # The isolated Python -S launcher omits setuptools' normal .pth activation.
    # Use its installed public package to supply distutils for vendor TVM on3.12.
    import setuptools  # noqa: F401
    sys.argv = [str(ROOT / 'tools/baseline_old1076_tla_cache_v2.py'), target, *args]
    runpy.run_path(sys.argv[0], run_name='__main__')
else:
    raise ValueError('No validated remote environment selection for ' + library)
