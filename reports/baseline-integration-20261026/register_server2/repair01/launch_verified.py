"""Run inside npu-guard: CPU prepare -> exec fresh runtime -> ordinary caller.
Usage: python3 -S -B launch_verified.py GROUP CASE RUN_DIR -- SCRIPT [args...]
No registry monkeypatch. Never use this command to replay accepted receipts.
"""
import json
import os
from pathlib import Path
import runpy
import sys
sys.dont_write_bytecode = True
from validated_runtime import MANIFEST, verify_pins, require_guard, check_environment, load

def prepare(group, cid, run, command):
    if group not in MANIFEST['groups'] or MANIFEST['contracts'].get(cid, {}).get('group') != group:
        raise ValueError('Only verified keys are supported')
    if any(name.split('.')[0] in ('torch', 'torch_npu', 'acl', 'numpy', 'tensorflow') for name in sys.modules):
        raise RuntimeError('Launch preparation must precede framework imports')
    verify_pins(group)
    require_guard(group)
    cfg = MANIFEST['groups'][group]
    run = Path(run).resolve()
    if not run.is_relative_to(Path(MANIFEST['run_root'])) or run.name != cid:
        raise ValueError('Use RUN_ROOT/<new-attempt>/<exact-case-id>')
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(cfg['environment'])
    env['TRANSFORMER_RUN_DIR'] = str(run)
    caches = ['TMPDIR','PIP_CACHE_DIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME',
              'ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH','ASCEND_CACHE_PATH',
              'TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','CCACHE_DIR']
    for name in caches:
        (run / name).mkdir()
        env[name] = str((Path('/proc') / str(os.getpid()) / 'cwd' / name) if group == 'attention' else (run / name))
    if group == 'gdn':
        env['GDN_RUN_DIR'] = str(run)
        load('resource_gate').require_capacity(Path(cfg['root']) / 'chunk_package', run)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    args = [cfg['executable']]
    if group == 'gdn':
        args += ['-S', '-B']
    args += [str(Path(__file__).resolve()), '--child', group, cid, '--', *command]
    return args, env, Path(cfg['cwd']) if group == 'gdn' else run

def main():
    args = sys.argv[1:]
    if args[0] == '--child':
        _, group, cid, sep, *command = args
        if sep != '--' or not command:
            raise ValueError('SCRIPT required')
        cfg = MANIFEST['groups'][group]
        sys.path[:0] = cfg['python_paths']
        check_environment(group)
        require_guard(group)
        if group == 'gdn':
            import setuptools
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
    else:
        group, cid, run, sep, *command = args
        if sep != '--' or not command:
            raise ValueError('SCRIPT required')
        argv, env, cwd = prepare(group, cid, run, command)
        os.chdir(cwd)
        os.execve(argv[0], argv, env)

if __name__ == '__main__':
    main()
