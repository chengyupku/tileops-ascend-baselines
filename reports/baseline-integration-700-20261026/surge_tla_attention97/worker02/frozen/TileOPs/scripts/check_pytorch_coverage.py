#!/usr/bin/env python3
"""Capture original NPU inputs and PyTorch outputs, then compare on isolated CPU.

capture: run only inside npu-guard; original manifest sizes and complete outputs.
compare: python -I -S -B; reads saved original inputs and independent CPU oracle.
list: CPU-only case selection. Each worker owns its --root directory.

Example:
  npu-guard run --owner torch355/gdn --wait 7200 --timeout 300 -- \
    .venv/bin/python TileOPs/scripts/check_pytorch_coverage.py capture \
    --root PATH --cases-file CASES --first --max-seconds 220
  python3 -I -S -B TileOPs/scripts/check_pytorch_coverage.py compare \
    --root PATH --cases-file CASES --skip-existing

Successful capture is not correctness acceptance. CPU compare must pass, and the
outer guard must exit successfully. Performance is not measured by these modes.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.abc
import importlib.util
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
TILEOPS = ROOT / 'TileOPs'
SCHEMA = 'pytorch-coverage-original-input/v2'
BOOT_SOURCES = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()


def guard_metadata():
    """Require the guarded foreground process group before importing torch_npu."""
    import fcntl
    path = Path('/run/lock/ascend-npu/device-0.lock')
    with path.open() as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(f, fcntl.LOCK_UN)
            raise RuntimeError('Device command must run inside npu-guard')
        meta = json.load(f)
    child = meta.get('child', {}).get('pid')
    if not child or os.getpgrp() != os.getpgid(child) or meta.get('phase') != 'running':
        raise RuntimeError('Current process is not in the active npu-guard foreground group')
    return meta


def bootstrap(cpu):
    os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0', FLA_NPU_DISABLE_PTH='1')
    sys.path[:0] = [str(TILEOPS), str(TILEOPS / 'src'), str(ROOT)]
    if not cpu:
        guard_metadata()
        return
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise RuntimeError('CPU modes require python -I -S -B (no .pth/backend startup)')
    sys.path.append(str(ROOT / '.venv/lib/python3.12/site-packages'))
    blocked = {'torch_npu', 'acl', 'aclnn', 'pyacl', 'torchair', 'triton', 'tilelang',
               'tvm', 'vllm', 'vllm_ascend', 'fla', 'cann', 'tbe', 'te', 'ascend'}

    class DenyDevice(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in blocked:
                raise ImportError('CPU checker forbids runtime: ' + fullname)

    sys.meta_path.insert(0, DenyDevice())
    find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda n, *a, **kw: (
        None if n.split('.')[0] in blocked else find_spec(n, *a, **kw))

    def audit(event, values):
        if event == 'ctypes.dlopen' and values and values[0]:
            name = str(values[0]).lower()
            if any(t in name for t in ('ascend', 'libacl', 'torch_npu', 'davinci', 'libcann', 'libdrv', 'libdsmi', 'libaicpu')):
                raise RuntimeError('Forbidden CPU runtime library: ' + name)
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            name = os.fsdecode(values[0])
            if name.startswith(('/dev/davinci', '/dev/devmm', '/dev/hisi', '/dev/nvidia')):
                raise RuntimeError('Forbidden CPU device file: ' + name)
    sys.addaudithook(audit)


def sources(op):
    from benchmarks.pytorch_coverage import module_for
    module = module_for(op)
    paths = {Path(__file__), Path(module.__file__), TILEOPS / 'benchmarks/pytorch_coverage/__init__.py',
             TILEOPS / 'benchmarks/api.py', TILEOPS / 'benchmarks/baselines.py'}
    paths.update((TILEOPS / 'workloads').rglob('*.py'))
    paths.update((TILEOPS / 'benchmarks/_cases').glob('*.py'))
    paths.update((TILEOPS / 'src/tileops/manifest').rglob('*.py'))
    paths.update((TILEOPS / 'src/tileops/manifest').rglob('*.yaml'))
    if 'pytorch_missing' in str(module.__file__):
        paths.update((TILEOPS / 'benchmarks/pytorch_missing').glob('*.py'))
    if BOOT_SOURCES is None:
        raise RuntimeError('Source versions must be frozen before importing workload/candidate modules')
    expected = {}
    for p in sorted(paths):
        key = str(p.relative_to(ROOT))
        if key not in BOOT_SOURCES:
            raise RuntimeError('Source appeared after process startup: ' + key)
        expected[key] = BOOT_SOURCES[key]
    check_sources(expected)
    return expected


def freeze_sources():
    """Freeze before imports, so cached code cannot acquire later file hashes."""
    paths = {Path(__file__), TILEOPS / 'benchmarks/api.py', TILEOPS / 'benchmarks/baselines.py'}
    for folder, patterns in ((TILEOPS / 'workloads', ('*.py',)),
                             (TILEOPS / 'benchmarks/_cases', ('*.py',)),
                             (TILEOPS / 'benchmarks/pytorch_coverage', ('*.py',)),
                             (TILEOPS / 'benchmarks/pytorch_missing', ('*.py',)),
                             (TILEOPS / 'src/tileops/manifest', ('*.py', '*.yaml'))):
        for pattern in patterns:
            paths.update(folder.rglob(pattern))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def install_source_fence():
    """Compile frozen project source bytes directly, bypassing timestamp-only pyc caches."""
    class FrozenLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = Path(self.path)
            content = path.read_bytes()
            key = str(path.relative_to(ROOT))
            if hashlib.sha256(content).hexdigest() != BOOT_SOURCES[key]:
                raise RuntimeError('Source changed before module import: ' + key)
            return compile(content, str(path), 'exec', dont_inherit=True)

    class FrozenFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith('.py'):
                source = Path(spec.origin).resolve()
                if source.is_relative_to(ROOT) and str(source.relative_to(ROOT)) in BOOT_SOURCES:
                    spec.loader = FrozenLoader(fullname, str(source))
                    return spec
            return None
    sys.meta_path.insert(0, FrozenFinder())


def check_sources(expected):
    changed = [p for p, h in expected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    if changed:
        raise RuntimeError('Capture source version changed: ' + ', '.join(changed))


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, tuple):
        return tuple(cpu_copy(x) for x in value)
    if isinstance(value, list):
        return [cpu_copy(x) for x in value]
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    return value


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from tensors(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)


def exact_bits(a, b):
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    aa, bb = a.contiguous().view(torch.uint8).reshape(-1), b.contiguous().view(torch.uint8).reshape(-1)
    return all(torch.equal(x, y) for x, y in zip(aa.split(1 << 24), bb.split(1 << 24), strict=True))


def mutation_layout(case):
    from benchmarks.pytorch_coverage import mutated_positions
    declarations = list(case._call.signature.inputs.items())
    if any(not isinstance(t.mutated, bool) for _, t in declarations):
        raise ValueError('Conditional mutation requires an explicit evaluated contract')
    declared = tuple(i for i, (_, t) in enumerate(declarations) if t.mutated)
    if mutated_positions(case) != declared:
        raise AssertionError('Candidate mutation declaration differs from manifest')
    # Benchmark convention: named outputs followed by complete mutated input states.
    offset = len(case._call.signature.outputs)
    return {i: offset + j for j, i in enumerate(declared)}


def check_tensor_contract(case, name, value, *, device, input_tensor):
    import torch
    spec = case._call.specs[name]
    if spec is None:
        assert value is None, f'Absent tensor {name} must be None'
        return
    assert isinstance(value, torch.Tensor), f'{name} must be a tensor'
    assert tuple(value.shape) == tuple(spec.shape), f'{name} shape {tuple(value.shape)} != {spec.shape}'
    assert value.dtype == getattr(torch, spec.dtype), f'{name} dtype {value.dtype} != {spec.dtype}'
    place = 'cpu' if device == 'cpu' or spec.cpu else device
    assert value.device.type == place, f'{name} device {value.device} != {place}'
    declarations = case._call.signature.inputs if input_tensor else case._call.signature.outputs
    declaration = declarations[name]
    if declaration.contiguous:
        assert value.is_contiguous(), f'{name} must be contiguous'
    if input_tensor and spec.values is not None:
        expected = torch.tensor(spec.values, dtype=value.dtype).reshape(spec.shape)
        assert exact_bits(value.detach().cpu(), expected), f'{name} declared metadata differs'


def check_inputs(case, inputs, *, device):
    names = list(case._call.signature.inputs)
    assert isinstance(inputs, (tuple, list)) and len(inputs) == len(names), 'Input arity differs from manifest'
    for name, value in zip(names, inputs, strict=True):
        check_tensor_contract(case, name, value, device=device, input_tensor=True)


def check_outputs(case, outputs, *, device):
    values = tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)
    declared = list(case._call.signature.outputs)
    mapping = mutation_layout(case)
    assert len(values) == len(declared) + len(mapping), 'Returned output/state arity differs from manifest'
    for name, value in zip(declared, values[:len(declared)], strict=True):
        check_tensor_contract(case, name, value, device=device, input_tensor=False)
    input_names = list(case._call.signature.inputs)
    for input_index, output_index in mapping.items():
        # Updated metadata may differ from original values, so only the storage contract applies.
        name = input_names[input_index]
        spec = case._call.specs[name]
        value = values[output_index]
        import torch
        assert isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(spec.shape)
        assert value.dtype == getattr(torch, spec.dtype) and value.device.type == device
    return mapping


def valid_capture(info, case):
    if info.get('schema') != SCHEMA or info.get('status') != 'captured':
        raise ValueError('Not a current capture')
    assert (info.get('op'), info.get('case_id')) == (case._call.signature.name, case.id)
    assert info['source_sha256'] == sources(case._call.signature.name), 'Capture differs from loaded process source versions'
    artifact = Path(info['artifact'])
    assert artifact.is_file() and artifact.stat().st_size == info['artifact_bytes'], 'Capture artifact missing/truncated'
    assert sha(artifact) == info['artifact_sha256'], 'Capture artifact hash mismatch'


def valid_comparison(row, info, case):
    valid_capture(info, case)
    assert row.get('status') == 'passed' and row.get('full_output_check') is True
    assert row.get('checker_sha256') == sha(__file__), 'Comparison uses an older checker'
    assert (row.get('op'), row.get('case_id')) == (case._call.signature.name, case.id)
    assert row['capture'] == info['artifact'] and row['capture_sha256'] == info['artifact_sha256']
    expected = Path(row['expected'])
    assert expected.is_file() and sha(expected) == row['expected_sha256'], 'Expected artifact missing/corrupted'
    check_sources(row['source_sha256'])


def case_dir(root, op, case_id):
    if '/' in op or '/' in case_id or case_id in ('.', '..'):
        raise ValueError('Unsafe case identifier')
    return root / op / case_id


def select(args):
    from benchmarks.api import cases
    from benchmarks.pytorch_coverage import registered
    keys = None
    if args.cases_file:
        keys = {tuple(x.strip().split('::', 1)) for x in args.cases_file.read_text().splitlines()
                if x.strip() and not x.lstrip().startswith('#')}
        if not keys or any(len(k) != 2 for k in keys):
            raise ValueError('Invalid or empty case list')
    ops = sorted({o for o, _ in keys} if keys else args.op or registered())
    if args.op:
        ops = [o for o in ops if o in args.op]
    selected = []
    matched = set()
    for op in ops:
        cs = [c for c in cases(op) if (keys is None or (op, c.id) in keys)
              and (not args.case or c.id == args.case)]
        matched.update((op, c.id) for c in cs)
        selected.extend(cs[:1] if args.first else cs)
    expected = {(o, c) for o, c in keys or matched if o in ops and (not args.case or c == args.case)}
    if matched != expected:
        raise ValueError('Case list differs from manifest: ' + repr(expected - matched))
    return selected


def capture(case, directory, args):
    import torch
    from benchmarks.pytorch_coverage import bind, make_inputs, mutated_positions, module_for
    op = case._call.signature.name
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    attempt = directory / stamp
    attempt.mkdir(parents=True)
    before_sources = sources(op)
    seed = int.from_bytes(hashlib.sha256(f'{op}::{case.id}'.encode()).digest()[:4], 'little')
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    start = time.monotonic()
    inputs = make_inputs(case, 'npu')
    check_inputs(case, inputs, device='npu')
    # Save original values before any allowed mutation. CPU-declared metadata remains CPU.
    original = cpu_copy(inputs)
    versions = [getattr(x, '_version', None) for x in inputs]
    fn = bind(case)
    declared_mutations = mutated_positions(case)
    torch.npu.synchronize()
    run_start = time.monotonic()
    with torch.no_grad():
        got = fn(*inputs)
    torch.npu.synchronize()
    elapsed = time.monotonic() - run_start
    mapping = check_outputs(case, got, device='npu')
    output = cpu_copy(got)
    mutation_outputs = {}
    for i, (old, current) in enumerate(zip(original, inputs, strict=True)):
        if not isinstance(current, torch.Tensor):
            continue
        after = cpu_copy(current)
        if i in declared_mutations:
            mutation_outputs[i] = after
        else:
            assert current._version == versions[i], f'Unexpected input mutation index {i}'
            assert exact_bits(old, after), f'Input values changed at index {i}'
    torch.npu.synchronize()
    check_sources(before_sources)
    packet = dict(schema=SCHEMA, op=op, case_id=case.id,
                  inputs=original, produced=output, mutated_after=mutation_outputs,
                  parameters=case._call.params, indices=case._call.indices)
    torch.save(packet, attempt / 'capture.pt')
    info = dict(schema=packet['schema'], status='captured', op=op, case_id=case.id,
                source_sha256=before_sources, runner_sha256=sha(__file__), seed=seed,
                candidate_module=module_for(op).__name__, guard=guard_metadata(),
                input_factory='module.make_inputs' if hasattr(module_for(op), 'make_inputs') else 'original Case.inputs',
                execution_seconds=elapsed, wall_seconds=time.monotonic()-start,
                artifact=str((attempt / 'capture.pt').resolve()), artifact_bytes=(attempt / 'capture.pt').stat().st_size,
                artifact_sha256=sha(attempt / 'capture.pt'), mutation_output_indices=mapping,
                input_devices=[str(x.device) if isinstance(x, torch.Tensor) else None for x in inputs],
                numerical_output_tensors=sum(1 for _ in tensors(output)),
                complete_output_elements=sum(t.numel() for t in tensors(output)),
                mutated_positions=list(declared_mutations), nonmutated_inputs_bitexact=True,
                torch_version=torch.__version__, recorded_at=datetime.now(timezone.utc).isoformat(),
                performance_measurement=False)
    check_sources(before_sources)
    write_json(attempt / 'capture.json', info)
    write_json(directory / 'latest_capture.json', info)
    return info


def compare(case, directory, args):
    import torch
    from benchmarks.pytorch_coverage import cpu_reference, mutated_positions
    from workloads.numerics import compare_outputs
    info = json.loads((directory / 'latest_capture.json').read_text())
    valid_capture(info, case)
    artifact = Path(info['artifact'])
    packet = torch.load(artifact, mmap=True, weights_only=False, map_location='cpu')
    assert packet['schema'] == SCHEMA
    assert (packet['op'], packet['case_id']) == (case._call.signature.name, case.id)
    inputs = packet['inputs']
    assert packet['parameters'] == case._call.params and packet['indices'] == case._call.indices, 'Packet parameters differ from manifest'
    check_inputs(case, inputs, device='cpu')
    mapping = check_outputs(case, packet['produced'], device='cpu')
    assert set(packet['mutated_after']) == set(mapping), 'Missing or unexpected captured mutation state'
    # Case.verification and custom references must see the same saved original inputs.
    case.__dict__['inputs'] = inputs
    case.__dict__.pop('verification', None)
    positions = mutated_positions(case)
    reference_inputs = tuple(x.clone() if i in positions and isinstance(x, torch.Tensor) else x
                             for i, x in enumerate(inputs))
    with torch.no_grad():
        expected = cpu_reference(case, reference_inputs)
    check_outputs(case, expected, device='cpu')
    assert all(t.device.type == 'cpu' for t in tensors(expected))
    evidence = case.verification
    if evidence.kind not in ('exact', 'custom'):
        raise ValueError('Complete coverage requires Exact/Custom evidence, got ' + evidence.kind)
    result = compare_outputs(packet['produced'], expected, evidence)
    if getattr(evidence, 'probe', None) is not None:
        raise ValueError('This workload requires a device probe; add an explicit capture probe before admission')
    expected_top = tuple(expected) if isinstance(expected, (tuple, list)) else (expected,)
    numerical_outputs = sum(bool(list(tensors(v))) for v in expected_top)
    assert result.checked_outputs == numerical_outputs > 0
    produced_top = tuple(packet['produced']) if isinstance(packet['produced'], (tuple, list)) else (packet['produced'],)
    for i, actual in packet['mutated_after'].items():
        if not exact_bits(actual, produced_top[mapping[i]]):
            raise AssertionError(f'Mutated input {i} differs from its returned output {mapping[i]}')
    rejected_controls = []
    for control in evidence.controls:
        with torch.no_grad():
            faulty = control.run(lambda *xs: cpu_reference(case, xs), inputs)
        try:
            compare_outputs(faulty, expected, evidence)
        except AssertionError:
            rejected_controls.append(control.name)
        else:
            raise AssertionError('Negative control accepted: ' + control.name)
        del faulty
    check_sources(info['source_sha256'])
    comparison_dir = artifact.parent / ('compare-' + uuid.uuid4().hex)
    comparison_dir.mkdir()
    expected_path = comparison_dir / 'expected.pt'
    torch.save(expected, expected_path)
    row = dict(op=packet['op'], case_id=packet['case_id'], status='passed',
               check=asdict(result), full_output_check=True, evidence_kind=evidence.kind,
               evidence_description=getattr(evidence, 'reason', None),
               complete_output_elements=sum(t.numel() for t in tensors(expected)),
               source_sha256=info['source_sha256'], capture=str(artifact), capture_sha256=sha(artifact),
               checker_sha256=sha(__file__), schema=SCHEMA, cpu_torch_version=torch.__version__,
               expected=str(expected_path), expected_sha256=sha(expected_path),
               original_npu_inputs=True, numerical_output_count=numerical_outputs,
               rejected_negative_controls=rejected_controls,
               structural_outputs=len(expected_top), guard=info['guard'],
               recorded_at=datetime.now(timezone.utc).isoformat(),
               outer_guard_success_requires_review=True)
    write_json(comparison_dir / 'comparison.json', row)
    write_json(directory / 'latest_comparison.json', row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('capture', 'compare', 'list'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cases-file', type=Path)
    p.add_argument('--op', action='append')
    p.add_argument('--case')
    p.add_argument('--first', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--max-seconds', type=float, default=0, help='Soft batch time limit between cases')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    bootstrap(args.mode != 'capture')
    global BOOT_SOURCES
    BOOT_SOURCES = freeze_sources()
    install_source_fence()
    import torch
    from workloads.device import set_run_device
    torch.set_num_threads(args.threads)
    if args.mode == 'capture':
        import torch_npu
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = False
        torch.npu.matmul.allow_hf32 = False
        set_run_device('npu')
    else:
        assert torch.__version__.endswith('+cpu')
        set_run_device('cpu')
    args.root.mkdir(parents=True, exist_ok=True)
    selected = select(args)
    if args.mode == 'list':
        for case in selected:
            print(case._call.signature.name + '::' + case.id)
        return 0
    started = time.monotonic()
    done = failures = 0
    for case in selected:
        op = case._call.signature.name
        directory = case_dir(args.root, op, case.id)
        if args.mode == 'compare' and not (directory / 'latest_capture.json').exists():
            continue
        target = directory / ('latest_capture.json' if args.mode == 'capture' else 'latest_comparison.json')
        if args.skip_existing and target.exists():
            old = json.loads(target.read_text())
            if old.get('status') in ('captured', 'passed'):
                try:
                    if args.mode == 'capture':
                        valid_capture(old, case)
                    else:
                        valid_comparison(old, json.loads((directory / 'latest_capture.json').read_text()), case)
                    continue
                except (RuntimeError, ValueError, AssertionError, OSError, KeyError):
                    pass
        if args.limit and done >= args.limit:
            break
        if done and args.max_seconds and time.monotonic()-started >= args.max_seconds:
            break
        print(json.dumps(dict(event='start', mode=args.mode, op=op, case_id=case.id)), flush=True)
        # Scalar constructor arguments must not regenerate a second random input set.
        if hasattr(case.workload, 'call'):
            case.workload.arguments = lambda c=case: c._call.arguments({})
        try:
            row = capture(case, directory, args) if args.mode == 'capture' else compare(case, directory, args)
        except Exception as exc:
            failures += 1
            row = dict(op=op, case_id=case.id, status='failed', error=f'{type(exc).__name__}: {exc}',
                       traceback=traceback.format_exc(), recorded_at=datetime.now(timezone.utc).isoformat())
        append(args.root / (args.mode + '.jsonl'), row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('source_sha256', 'traceback', 'guard')}, ensure_ascii=False), flush=True)
        done += 1
        case._release()
        gc.collect()
        if args.mode == 'capture':
            torch.npu.synchronize()
            torch.npu.empty_cache()
    print(json.dumps(dict(mode=args.mode, executed=done, failed=failures, seconds=time.monotonic()-started)), flush=True)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
