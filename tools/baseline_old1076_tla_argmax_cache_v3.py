#!/usr/bin/env python3
"""Select an attempt-local JIT cache using the original public runtime API.

This launcher imports device runtimes: invoke only inside the foreground guard.
The vendor disk loader does not restore adapter.libpath; a new cache avoids that
loader without changing compilation, generated kernels, or evidence checking.
Keep process-local caching enabled and retain every generated artifact.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    target, *args = sys.argv[1:]
    if Path(target).resolve() != ROOT / 'tools/baseline_old1076_tla_argmax_v3.py':
        raise ValueError('Only the versioned original-operator composition entrypoint is allowed')
    out = Path(args[args.index('--out') + 1]).resolve()
    cache = out / 'jit_kernel_cache'
    cache.mkdir(exist_ok=False)
    sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT),
                    str(ROOT / 'reports/baseline-cache-20261027/python')]
    from benchmarks.ascend_all_pairs.tla.native import runtime
    tl = runtime()
    tl.set_cache_dir(str(cache))
    if Path(tl.get_cache_dir()).resolve() != cache:
        raise RuntimeError('Original public cache directory selection failed')
    receipt = out / 'launch_environment.json'
    receipt.write_text(json.dumps({
        'selection': 'original tilelang-ascend public set_cache_dir; fresh attempt directory; memory cache enabled',
        'jit_cache': str(cache),
        'launcher': str(Path(__file__).resolve()),
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    os.environ['BASELINE_CACHE_LAUNCH_RECEIPT'] = str(receipt)
    sys.argv = [target, *args]
    runpy.run_path(target, run_name='__main__')


if __name__ == '__main__':
    main()
