#!/usr/bin/env python3
"""Full canonical correctness, then the existing isolated FFTS benchmark core.

Invoked ONLY inside npu-guard. This file changes no baseline or validator.
The supervisor adds the final guard receipt before publishing reusable results.
"""
import argparse
import ctypes
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WRAPPER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]
sys.path.insert(0, str(ROOT / 'reports/baseline-cache-20261027/python'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(',', ':')).encode()).hexdigest()


def put(path, row):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def native_fingerprints(paths, cache_dir):
    """Hash each unchanged installed binary once, not once per pair."""
    path = cache_dir / 'native_hashes.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    out = {}
    for name in sorted(paths):
        p = Path(name)
        if not p.is_file():
            continue
        s = p.stat()
        stamp = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        item = old.get(name, {})
        if item.get('stamp') != stamp:
            with p.open('rb') as f:
                h = hashlib.sha256()
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(block)
            item = {'stamp': stamp, 'sha256': h.hexdigest()}
            old[name] = item
        out[name] = item['sha256']
    put(path, old)
    return out


def install_exported_flush(torch):
    """Deploy the identical exported flush kernel on a host without TileLang JIT.

No source/schedule changes: the checked-in generated 1GiB kernel is compiled
for the remote host ABI; measurement still uses the unmodified unified core,
including its profiled marker validation and per-invocation flush/sync.
"""
    directory = ROOT / 'reports/baseline-cache-20261027/flush'
    library = directory / 'flush.so'
    if not library.exists():
        return None
    source = json.loads((directory / 'SOURCE.json').read_text())
    if hashlib.sha256((directory / 'host_kernel.asc').read_bytes()).hexdigest() != source['generated_source_sha256']:
        raise ValueError('Exported flush source changed')
    from tools import measure_baselines as core
    dll = ctypes.CDLL(str(library))
    dll.init.argtypes, dll.init.restype = [ctypes.c_char_p], ctypes.c_int
    dll.call.argtypes, dll.call.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int
    dll.get_last_error.argtypes, dll.get_last_error.restype = [], ctypes.c_char_p
    if hashlib.sha256((directory / 'flush.aibin').read_bytes()).hexdigest() != source['device_binary_sha256']:
        raise ValueError('Exported flush device binary changed')
    if dll.init(str(directory / 'flush.aibin').encode()) != 0:
        raise RuntimeError('Exported original flush initialization failed')
    def flush(buffer):
        rc = dll.call(buffer.data_ptr(), torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(str(dll.get_last_error()))
    buffer = torch.ones(268435456, dtype=torch.int32, device='npu')
    flush(buffer)
    torch.npu.synchronize()
    if torch.count_nonzero(buffer).item() != 0:
        raise ValueError('Original exported flush did not cover the full 1GiB buffer')
    core._flush_kernel_cache[torch.npu.current_device()] = (flush, buffer)
    return {**source, 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'whole_buffer_check': '268435456 int32 ones all overwritten with zero'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    task = json.loads(a.task.read_text())
    library, op, case_id = task['key']
    a.out.mkdir(parents=True, exist_ok=True)
    result = {'schema': 1, 'key': task['key'], 'task_id': task['id'], 'status': 'started',
              'started_at': time.time(), 'eligible_basis': task['eligible_basis']}
    phase = lambda name: put(a.out / 'phase.json', {'phase': name, 'at': time.time(), 'pid': os.getpid(), 'key': task['key']})
    captured = {}
    cleanup = None
    try:
        checker_path = ROOT / 'TileOPs/scripts/cache_frozen_pair_check.py'
        spec = importlib.util.spec_from_file_location('cache_canonical_checker', checker_path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        checker.require_guard()
        # Factory capture retains the exact callable/reset/arguments. Defer cleanup
        # until measurement, since the original checker normally releases handles.
        from benchmarks.ascend_all_pairs import registry
        original = registry.candidate_factory

        def factory(case, lib):
            make = original(case, lib)
            def build():
                candidate = make()
                captured.update(case=case, candidate=candidate)
                return dataclasses.replace(candidate, cleanup=None)
            return build

        registry.candidate_factory = factory
        receipt = a.out / 'correctness.json'
        sys.argv = [str(checker_path), '--op', op, '--case-id', case_id, '--library', library, '--output', str(receipt)]
        if op in ('GDNFwdOp', 'KDAFwdOp'):
            sys.argv.append('--cpu-fp64-reference')
        phase('full_correctness')
        rc = checker.main()
        check = json.loads(receipt.read_text())
        result['correctness_receipt'] = str(receipt)
        result['correctness_sha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        if 'candidate' in captured:
            cleanup = captured['candidate'].cleanup
        if rc or check.get('status') != 'verified':
            result.update(status='correctness_not_verified', checker_status=check.get('status'),
                          error=check.get('error', check.get('reason')), stage=check.get('stage'))
            if 'aclSetCompileopt' in str(check.get('error', '')):
                try:
                    acl = ctypes.CDLL('libascendcl.so')
                    acl.aclGetRecentErrMsg.restype = ctypes.c_char_p
                    result['native_error_detail'] = (acl.aclGetRecentErrMsg() or b'').decode(errors='replace')
                except Exception:
                    pass
            return 1
        snapshot = check['case_snapshot']
        expected = task['case']
        if (snapshot['parameters'] != expected['params'] or
                snapshot['declared_tensors'] != expected['tensors']):
            raise ValueError('Catalog case contract differs from canonical runtime; cache refused')
        if not check.get('inputs_preserved') or check.get('provider_source_changed') or check.get('oracle_source_changed'):
            raise ValueError('Full correctness/input/source invariants missing')
        import torch
        import torch_npu
        from tools.measure_baselines import _time_reference, _environment
        from tools.baseline_protocol import PROTOCOL, PROTOCOL_ID
        candidate, case = captured['candidate'], captured['case']
        impl = candidate.implementation
        inputs = case.inputs if impl.args is None else impl.args
        wall, child_calls = [], []
        active = [False]

        def audit(event, args):
            if active[0] and event in ('subprocess.Popen', 'os.fork', 'os.posix_spawn'):
                child_calls.append(event)
        sys.addaudithook(audit)

        def measured(*args):
            n = len(child_calls)
            start = time.perf_counter()
            active[0] = True
            try:
                out = impl.run(*args)
                torch.npu.synchronize()
            finally:
                active[0] = False
            wall.append({'ms': (time.perf_counter() - start) * 1000, 'child': len(child_calls) != n})
            return out

        phase('profile_3_warmup_15_repeats')
        if library != 'ops-math':
            raise ValueError('This versioned measurement wrapper owns ops-math only')
        import subprocess
        from benchmarks.ascend_all_pairs.ops_math.specs import VENDOR
        def map_inputs(value):
            if isinstance(value, torch.Tensor): return value.detach().cpu()
            if isinstance(value, tuple): return tuple(map_inputs(x) for x in value)
            if isinstance(value, list): return [map_inputs(x) for x in value]
            if isinstance(value, dict): return {k:map_inputs(v) for k,v in value.items()}
            return value
        transport = a.out / 'profile_inputs.pt'
        profile_receipt = a.out / 'native_profile.json'
        torch.save({'op':op, 'case_id':case_id, 'params':case.params,
                    'inputs':map_inputs(inputs)}, transport)
        child_env = dict(os.environ)
        child_env['ASCEND_CUSTOM_OPP_PATH'] = str(VENDOR)
        child_env['PYTHONDONTWRITEBYTECODE'] = '1'
        command = [sys.executable, '-B', str(ROOT/'tools/baseline_support_math_profile_child_v6.py'),
                   '--input', str(transport), '--output', str(profile_receipt)]
        with (a.out/'native_profile.log').open('w') as log:
            rc = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT).returncode
        transport.unlink(missing_ok=True)
        if rc:
            raise RuntimeError('Original native profiler child failed: '+str(a.out/'native_profile.log'))
        profile = json.loads(profile_receipt.read_text())
        if (profile['status'] != 'profiled_original_native' or profile['key'] != task['key']
                or profile['process_group'] != os.getpgrp() or profile['profiled_calls'] != 19):
            raise ValueError('Native profiler identity/key/coverage mismatch')
        timing, wall = profile['timing'], profile['wall']
        result['exported_flush'] = profile['exported_flush']
        result['native_profile_receipt'] = str(profile_receipt)
        result['native_profile_sha256'] = hashlib.sha256(profile_receipt.read_bytes()).hexdigest()
        candidate = dataclasses.replace(candidate, timing_mode=profile['timing_mode'])
        if len(timing['latency_samples_ms']) != 15 or any(x <= 0 for x in timing['latency_samples_ms']):
            raise ValueError('Incomplete/nonpositive 15-repeat profile')
        if len(wall) != 19 or any(x['child'] for x in wall[-15:]):
            raise ValueError('Subprocess kernel coverage is not guaranteed by parent profiler; do not cache partial device time')
        result.update(timing=timing, end_to_end_samples_ms=[x['ms'] for x in wall[-15:]],
                      end_to_end_mean_ms=statistics.mean(x['ms'] for x in wall[-15:]),
                      timing_mode=candidate.timing_mode,
                      comparison_metric='device_latency_ms' if candidate.timing_mode == 'device_async' else 'end_to_end_mean_ms')
        phase('cleanup_and_provenance')
        torch.npu.synchronize()
        if cleanup:
            cleanup()
            cleanup = None
        sources = {}
        for field in ('source_sha256', 'provider_source_sha256', 'oracle_source_sha256'):
            sources.update(check.get(field, {}))
        sources.update(profile['source_sha256'])
        sources[str(profile_receipt)] = result['native_profile_sha256']
        launch_receipt = os.environ.get('BASELINE_CACHE_LAUNCH_RECEIPT')
        if launch_receipt:
            launch_path = Path(launch_receipt).resolve()
            if launch_path.parent != a.out.resolve():
                raise ValueError('Launch environment receipt must belong to this attempt')
            launch = json.loads(launch_path.read_text())
            sources[launch['launcher']] = launch['launcher_sha256']
            sources[str(launch_path)] = hashlib.sha256(launch_path.read_bytes()).hexdigest()
        for relative, before in sources.items():
            path = Path(relative)
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != before:
                raise ValueError('Source or native component changed: ' + relative)
        env = _environment(torch, torch_npu)
        env.pop('timestamp_utc', None)
        # Build identities, not only versions: preserve installed binary fingerprints.
        libs = set()
        for line in Path('/proc/self/maps').read_text().splitlines():
            path = line.split()[-1]
            if path.startswith('/') and '.so' in path and any(s in path for s in ('Ascend', 'ascend', 'torch_npu', 'tilelang', 'tvm', 'libcust')):
                libs.add(path)
        env['loaded_native_sha256'] = {**native_fingerprints(libs, a.task.resolve().parents[2]), **profile['environment']['loaded_native_sha256']}
        env['native_profile_process'] = {k:v for k,v in profile['environment'].items() if k != 'loaded_native_sha256'}
        protocol = {**PROTOCOL, 'id': PROTOCOL_ID, 'core_sha256': hashlib.sha256((ROOT / 'tools/measure_baselines.py').read_bytes()).hexdigest()}
        identity = {'case': expected, 'library': library, 'sources': sources, 'environment': env,
                    'protocol': protocol, 'verifier_sha256': check['verifier_sha256'],
                    'wrapper_sha256': WRAPPER_SHA256, 'exported_flush': result['exported_flush'],
                    'adapter_sha256': check['adapter_sha256'], 'registry_sha256': check['providers_sha256']}
        # Library-specific binaries belong in the cache key; shared runtime
        # versions define which different libraries can be compared together.
        cohort = {k: env.get(k) for k in ('hostname', 'platform', 'torch', 'torch_npu', 'device_name', 'cann_home', 'ascend_npu_arch', 'soc_version')}
        cohort['shared_native'] = {p:h for p,h in env['loaded_native_sha256'].items() if '/driver/' in p}
        result.update(status='measured_pending_guard', cache_key=digest(identity), identity=identity,
                      cohort_key=digest({'environment': cohort, 'protocol': protocol}),
                      performance_valid=False)
        return 0
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        return 1
    finally:
        if cleanup:
            try:
                cleanup()
            except Exception as exc:
                result.update(status='cleanup_failed', cleanup_error=str(exc))
        result['finished_at'] = time.time()
        put(a.out / 'measurement.json', result)
        phase(result['status'])


if __name__ == '__main__':
    raise SystemExit(main())
