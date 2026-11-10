"""Future authorized child entry. Never executed by this CPU-only worker."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from direct_acl import require_guard

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def main():
    require_guard()
    raise RuntimeError("Known unchanged DeltaNet canonical oracle SDK metadata blocker; see DELTA_ORACLE_PREFLIGHT.json; no framework initialization or original oracle retry")
    from resource_gate import require_capacity
    run_dir = Path(os.environ['GDN_RUN_DIR']).resolve()
    require_capacity(HERE, run_dir)
    recipe = json.loads((HERE / 'RUNTIME_RECIPE.json').read_text())
    if not recipe['dependency_closure_ready']:
        raise RuntimeError('Unresolved pinned 9.2 dependency closure; see HANDOFF.md')
    for pin in json.loads((HERE / 'RUNTIME_PINS.json').read_text())['files']:
        if hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() != pin['sha256']:
            raise RuntimeError('Pinned source/runtime changed: ' + pin['path'])
    for key, value in recipe['required_environment'].items():
        if os.environ.get(key) != value:
            raise RuntimeError('Required environment mismatch: ' + key)
    workspace = Path('/opt/tileops-all-pairs/workspace')
    sys.path[:0] = [str(HERE), str(workspace / 'TileOPs'),
                   str(workspace / 'TileOPs/src'), str(workspace)]
    sys.path.extend(recipe['python_paths'])
    import setuptools
    import benchmarks.ascend_all_pairs.registry as registry
    from binding import build
    original = registry.candidate_factory

    def factory(case, library):
        if library == 'ops-transformer' and case._call.signature.name in ('GDNFwdOp','DeltaNetFwdOp'):
            return lambda: build(case)
        return original(case, library)

    registry.candidate_factory = factory
    # This is the original all-output verifier, source-pinned without modification.
    # Its ops-transformer CPU-reference placement and tolerance remain unchanged.
    try:
        runpy.run_path(recipe['checker'], run_name='__main__')
    finally:
        maps = Path('/proc/self/maps').read_text()
        (run_dir / 'loaded-maps.txt').write_text(maps)
        identities = []
        for name in sorted(set(line.split()[-1] for line in maps.splitlines() if '/' in line)):
            path = Path(name)
            if path.is_file() and ('.so' in name):
                identities.append({'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        (run_dir / 'loaded-libraries.json').write_text(json.dumps(identities, indent=2))


if __name__ == '__main__':
    main()
