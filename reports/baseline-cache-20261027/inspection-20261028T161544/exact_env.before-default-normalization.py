"""CPU-only exact deployment dispatch; the entire invocation stays in npu-guard.

Reuse original deployment recipes without changing adapters or their admission
checks. This version is bound into new cache receipts and must remain immutable.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_suffix('.json')


def selection(task):
    route = json.loads(CONFIG.read_text()).get(task['id'])
    if route and (route['key'] != task['key'] or route['case'] != task['case']):
        raise ValueError('Exact deployment contract changed')
    return route


def guard_owner(task):
    with Path('/run/lock/ascend-npu/device-0.lock').open() as lock:
        meta = json.load(lock)
        child = meta['child']
        ticks = int(Path(f"/proc/{child['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19])
        expected = 'baseline-cache-20261027/local/' + task['id']
        if not (meta['phase'] == 'running' and meta['owner'] == expected
                and child['pid'] == os.getpgrp() and ticks == child['start_ticks']
                and meta['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
            raise RuntimeError('Live campaign guard identity required')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return expected
        fcntl.flock(lock, fcntl.LOCK_UN)
        raise RuntimeError('Campaign kernel lock is not held')


def main():
    if sys.argv[1] == '--preimport':
        # Normal site hooks remain unchanged. Re-select the original recipe's
        # OPP after site startup but strictly before any device runtime import.
        _, _, library, target, *args = sys.argv
        if any(name in sys.modules for name in ('torch', 'torch_npu', 'acl', 'triton', 'tilelang')):
            raise RuntimeError('Deployment selection is too late: a runtime is already imported')
        task = json.loads(Path(args[args.index('--task') + 1]).read_text())
        route = selection(task)
        if not route or route['kind'] != 'original_recipe':
            raise ValueError('Pre-import selection requires an original exact recipe')
        guard_owner(task)
        expected = os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR']
        before = os.environ.get('ASCEND_CUSTOM_OPP_PATH')
        os.environ['ASCEND_CUSTOM_OPP_PATH'] = expected
        receipt = Path(os.environ['BASELINE_CACHE_LAUNCH_RECEIPT'])
        evidence = json.loads(receipt.read_text())
        evidence['pre_device_import_opp'] = {'after_site': before, 'selected_original': expected,
                                           'runtime_modules_loaded': False}
        receipt.write_text(json.dumps(evidence, indent=2) + '\n')
        sys.argv = [target, *args]
        runpy.run_path(target, run_name='__main__')
        return
    library, target, *args = sys.argv[1:]
    task_path = Path(args[args.index('--task') + 1])
    out = Path(args[args.index('--out') + 1]).resolve()
    task = json.loads(task_path.read_text())
    route = selection(task)
    env = dict(os.environ)
    env['BASELINE_CACHE_EXACT_ENV_SELECTED'] = '1'
    if not route:
        os.execve('/bin/bash', ['bash', str(ROOT / 'tools/baseline_cache_python.sh'), library, target, *args], env)
    if Path(target).resolve() != ROOT / 'tools/measure_baseline_cache_pair.py':
        raise ValueError('Only canonical cache measurements are allowed')
    owner = guard_owner(task)
    receipt = out / 'launch_environment.json'
    evidence = {'selection': route, 'owner': owner,
                'launcher': str(Path(__file__).resolve()),
                'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest()}
    receipt.write_text(json.dumps(evidence, indent=2) + '\n')
    env['BASELINE_CACHE_LAUNCH_RECEIPT'] = str(receipt)
    python = str(ROOT / '.venv/bin/python')
    if route['kind'] == 'original_recipe':
        recipe = Path(route['recipe'])
        if hashlib.sha256(recipe.read_bytes()).hexdigest() != route['recipe_sha256']:
            raise ValueError('Original deployment recipe changed')
        env[route['owner_var']] = owner
        env[route['root_var']] = str(out / 'original_deployment_run')
        os.execve('/bin/bash', ['bash', str(recipe), python, str(Path(__file__).resolve()),
                              '--preimport', library, target, *args], env)
    if route['kind'] == 'classic_fft':
        # Use the recorded classic FFT vendor before importing any device runtime.
        # The unchanged provider validates the deployment and original dependencies.
        script = '''source "$1/env.sh"
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH" ASCEND_AICPU_PATH="$ASCEND_HOME_PATH" ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export ASCEND_CLASSIC_MATH_VENDOR="$1/.cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_CLASSIC_MATH_VENDOR"
export LD_LIBRARY_PATH="$ASCEND_CLASSIC_MATH_VENDOR/op_api/lib:${LD_LIBRARY_PATH:-}"
export FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1
shift
exec "$@"
'''
        os.execve('/bin/bash', ['bash', '-e', '-c', script, 'baseline-cache-fft', str(ROOT), python, target, *args], env)
    raise ValueError('Unknown deployment route')


if __name__ == '__main__':
    main()
