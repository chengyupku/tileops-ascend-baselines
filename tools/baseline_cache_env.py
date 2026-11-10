"""Select existing original per-case runtime deployments before Python startup.

Run with system Python -I -S. This selector imports only stdlib and the project's
existing CPU metadata helpers; it does not import device runtimes or factories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def selected_environment(task, base):
    env = dict(base)
    lib, op, case_id = task['key']
    evidence = {'key': task['key'], 'selector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if lib == 'ops-math':
        # Match the existing library's checker startup isolation. Its expanded
        # graph OPP belongs only to its guarded native child, not the oracle.
        graph = ROOT / '.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math'
        full = ROOT / '.cache/all-pairs/ops_math/install/vendors/allpairs_math_math'
        vendor = ROOT / '.tools/baselines/opp/ops-math/vendors/baseline_math_math'
        env['ASCEND_CUSTOM_OPP_PATH'] = str(vendor) if op == 'AddFwdOp' else ''
        previous = [p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                    if p and not any(p.startswith(str(v)) for v in (graph, full))]
        if op == 'AddFwdOp':
            if not (vendor / 'op_api/lib/libcust_opapi.so').is_file():
                raise FileNotFoundError(vendor)
            previous.insert(0, str(vendor / 'op_api/lib'))
        env['LD_LIBRARY_PATH'] = ':'.join(previous)
        evidence['selection'] = 'original Add OPP' if op == 'AddFwdOp' else 'original ops_math checker isolation; no expanded OPP in oracle'
    elif lib == 'ops-nn':
        here = ROOT / 'TileOPs/benchmarks/ascend_all_pairs/ops_nn'
        reviewed = json.loads((here / 'accepted_verified_20261026.json').read_text())['records']
        record = next((r for r in reviewed if r['key'] == task['key']), None)
        vendor = ROOT / '.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'
        if record:
            vendor = Path(record['vendor'])
            evidence['selection'] = 'existing accepted_verified_20261026 exact key'
        else:
            pinned = json.loads((here / 'accepted_pinned_cases.json').read_text())
            continuation = json.loads((here / 'accepted_continuation_cases.json').read_text())['records']
            continued = next((r for r in continuation if (r['op'], r['case_id']) == (op, case_id)), None)
            if continued:
                vendor = ROOT / continued['required_vendor']
                evidence['selection'] = 'existing continuation required_vendor'
            elif case_id in pinned.get(op, ()):
                evidence['selection'] = 'existing preserved install3 route'
            elif op in runpy.run_path(str(here / 'catalog.py'))['LEGACY']:
                vendor = ROOT / '.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn'
                evidence['selection'] = 'existing legacy GraphCall classic vendor'
            else:
                evidence['selection'] = 'existing install3 route'
        if not vendor.is_dir():
            raise FileNotFoundError(vendor)
        env['OPS_NN_VENDOR'] = env['ASCEND_CUSTOM_OPP_PATH'] = str(vendor)
        previous = [p for p in env.get('LD_LIBRARY_PATH', '').split(':')
                    if p and '/vendors/allpairs_nn/' not in p and '/vendors/classicpilot_nn/' not in p]
        env['LD_LIBRARY_PATH'] = ':'.join([str(vendor / 'op_api/lib'), *previous])
        evidence['vendor'] = str(vendor)
    elif lib == 'flash-linear-attention-npu':
        source = ROOT / 'TileOPs/benchmarks/ascend_all_pairs/fla/environment.py'
        helper = runpy.run_path(str(source))
        case = task['case']
        mode = helper['mode_for'](op, case['tensors'], case['params'])
        # The current Bmm provider uses original ChunkFwdO for K<=128 for
        # both FP16 and BF16, so its original GDN package must precede Torch.
        if op == 'BmmFwdOp' and case['tensors']['a'][0][-1] <= 128:
            mode = 'gdn'
        env = helper['configured_env'](mode, env)
        if env.get('ASCEND_CLASSIC_FLA_PACKAGE'):
            package = Path(env['ASCEND_CLASSIC_FLA_PACKAGE'])
            if not package.is_dir():
                raise FileNotFoundError(package)
            # The wheel's .pth hook otherwise adds its causal-only OPP before
            # the selected native package. All required paths are already set.
            env['FLA_NPU_DISABLE_PTH'] = '1'
        evidence.update(mode=mode, environment_helper_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        package=env.get('ASCEND_CLASSIC_FLA_PACKAGE'))
    else:
        raise ValueError('Selector does not own library: ' + lib)
    evidence['selected_environment'] = {k: env.get(k) for k in
        ('OPS_NN_VENDOR', 'ASCEND_CUSTOM_OPP_PATH', 'ASCEND_CLASSIC_FLA_PACKAGE', 'FLA_NPU_DISABLE_PTH')}
    return env, evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    task_path = Path(command[command.index('--task') + 1])
    out = Path(command[command.index('--out') + 1])
    env, evidence = selected_environment(json.loads(task_path.read_text()), os.environ)
    (out / 'environment_selection.json').write_text(json.dumps(evidence, indent=2) + '\n')
    os.execve(args.python, [args.python, *command], env)


if __name__ == '__main__':
    main()
