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

ROOT = Path('/opt/tileops-baseline-cache/workspace')
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
        if task.get('manifest_overlay') == 'fixed_v010':
            if not ((op == 'FusedAddLayerNormFwdOp' and library in {'vllm-ascend','sgl-kernel-npu','ops-math'}) or (op in {'GDNFwdOp','DeltaNetFwdOp','KDAFwdOp'} and library == 'vllm-ascend') or (op == 'GDNFwdOp' and library == 'sgl-kernel-npu')):
                raise ValueError('Unadmitted manifest overlay scope')
            import tileops.manifest as manifest
            overlay = Path(__file__).resolve().parent / 'fixed_v010_spec'
            expected = json.loads((overlay / 'SHA256.json').read_text())
            actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in overlay.glob('*.yaml')}
            if actual != expected:
                raise ValueError('Fixed manifest snapshot changed')
            manifest._SPEC_DIR = str(overlay)
            manifest.load_manifest.cache_clear()
            manifest.load_adts.cache_clear()
            result['manifest_overlay_sha256'] = actual
        checker_path = ROOT / 'TileOPs/scripts/cache_frozen_pair_check.py'
        spec = importlib.util.spec_from_file_location('cache_canonical_checker', checker_path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        checker.require_guard()
        # Factory capture retains the exact callable/reset/arguments. Defer cleanup
        # until measurement, since the original checker normally releases handles.
        from benchmarks.ascend_all_pairs import registry
        # Private top-level compatibility only; canonical checker and kernels unchanged.
        from benchmarks.ascend_all_pairs import tla
        private = Path(__file__).resolve().parent
        spec_adapter = importlib.util.spec_from_file_location('benchmarks.ascend_all_pairs.tla.adapters', private / 'tla_adapters_v2.py')
        adapter = importlib.util.module_from_spec(spec_adapter)
        sys.modules[spec_adapter.name] = adapter
        spec_adapter.loader.exec_module(adapter)
        tla.adapters = adapter
        base_factory = registry.candidate_factory
        if op in {'LerpScalarFwdOp','ClampTensorFwdOp','LogicalAndFwdOp','LogicalOrFwdOp'} and library == 'tilelang-ascend' and (op not in {'LogicalAndFwdOp','LogicalOrFwdOp'} or task['case']['params']['input'][1] != 'bool'):
            spec_composed = importlib.util.spec_from_file_location('flex_tla_composed_v2', private / 'tla_composed_v2.py')
            composed = importlib.util.module_from_spec(spec_composed)
            sys.modules[spec_composed.name] = composed
            spec_composed.loader.exec_module(composed)
            registry.candidate_factory = composed.factory
        if task.get('route') == 'original_tla_gemm_adapter':
            if library != 'tilelang-ascend' or op not in {'GemmFwdOp','BmmFwdOp'}:
                raise ValueError('Invalid explicit original GEMM route')
            # Reuse existing original callable; current outer hostguard already checked.
            def native_gemm_factory(case, lib):
                if lib != library or case._call.signature.name != op or case.id != case_id:
                    raise ValueError('Exact batch3 route mismatch')
                return lambda: tla._make(op, case)
            registry.candidate_factory = native_gemm_factory
        if task.get('route') == 'vllm_group1_im2col':
            if library != 'vllm-ascend' or op not in {'Conv2dFwdOp','Conv3dFwdOp'}:
                raise ValueError('Invalid private convolution route')
            spec = importlib.util.spec_from_file_location('flex_vllm_conv_v1', private / 'vllm_conv_v1.py')
            conv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conv)
            def conv_factory(case, lib):
                if lib != library or case.id != case_id or case._call.signature.name != op:
                    raise ValueError('Exact convolution route mismatch')
                return lambda: conv.make(case)
            registry.candidate_factory = conv_factory
        if task.get('route') == 'ops_math_add_layernorm':
            if library != 'ops-math' or op != 'FusedAddLayerNormFwdOp':
                raise ValueError('Invalid private ops-math route')
            sp = importlib.util.spec_from_file_location('flex_ops_math', private / 'ops_math_add_layernorm_v1.py')
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
            def math_factory(case, lib):
                if lib != library or case.id != case_id:
                    raise ValueError('Exact ops-math case mismatch')
                return lambda: mod.make(case)
            registry.candidate_factory = math_factory
        if task.get('route') == 'original_vllm_kda':
            if op != 'KDAFwdOp' or library != 'vllm-ascend':
                raise ValueError('Invalid exact original KDA route')
            from benchmarks.ascend_all_pairs.vllm.linear_attention import linear_attention as original_kda
            def kda_factory(case, lib):
                plain = lambda v: json.loads(json.dumps(v))
                if lib != library or case.id != case_id or plain(case.params) != task['case']['params'] or plain(case._call.tensors) != task['case']['tensors']:
                    raise ValueError('KDA current exact contract mismatch')
                if any(case.params.get(k) for k in ('use_gate_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval')) or case.params['q'][0][1] > 16384:
                    raise ValueError('Unadmitted KDA resource/call-count scope')
                return lambda: original_kda(case)
            registry.candidate_factory = kda_factory
        if task.get('route') == 'original_sgl_gdn_fp16':
            if op != 'GDNFwdOp' or library != 'sgl-kernel-npu':
                raise ValueError('Invalid original SGL FP16 GDN route')
            from benchmarks.ascend_all_pairs.sgl.linear import gdn as original_sgl_gdn
            def sgl_gdn_factory(case, lib):
                plain = lambda v: json.loads(json.dumps(v))
                p = case.params
                if lib != library or case.id != case_id or plain(p) != task['case']['params'] or plain(case._call.tensors) != task['case']['tensors']:
                    raise ValueError('SGL GDN current exact contract mismatch')
                if p['q'][1] != 'float16' or p['q'][0][1] > 16384 or (p['q'][0][3] != p['v'][0][3] or p['v'][0][2] % p['q'][0][2] != 0) or any(p.get(k) for k in ('use_gate_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval')):
                    raise ValueError('Unadmitted SGL FP16 scope')
                if os.environ.get('GDN_USE_MEGA_GDN','0') != '0':
                    raise ValueError('Require original default SGL complete pipeline')
                def make_sgl():
                    native = original_sgl_gdn(case)
                    original_run = native.implementation.run
                    def run(q, k, v, g, *rest):
                        # Native cumsum output is FP32; preserve exact FP16 gate values.
                        # Runtime conversion is part of this timed call, not precomputed.
                        repeats = v.shape[2] // q.shape[2]
                        if repeats != 1:
                            q, k = (x.repeat_interleave(repeats, dim=2) for x in (q, k))
                        return original_run(q, k, v, g.float(), *rest)
                    return dataclasses.replace(native,
                        implementation=dataclasses.replace(native.implementation, run=run),
                        source_sha256=native.source_sha256 | {str(Path(__file__).resolve()): WRAPPER_SHA256},
                        adaptation=native.adaptation + '; lossless gate FP16->FP32 and shared Q/K head materialization at native API boundary, conversions timed')
                return make_sgl
            registry.candidate_factory = sgl_gdn_factory
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

        phase('install_exported_flush')
        result['exported_flush'] = install_exported_flush(torch)
        phase('profile_3_warmup_15_repeats')
        timing = _time_reference(torch, torch_npu, measured, inputs, warmup=3, repeats=15, reset=impl.reset)
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
        hal = ctypes.CDLL('/usr/local/Ascend/driver/lib64/driver/libascend_hal.so')
        for label, module, expected_cores in [('aicore_count',4,32),('vectorcore_count',7,64)]:
            count = ctypes.c_int64(0)
            rc = hal.halGetDeviceInfo(0,module,3,ctypes.byref(count))
            if rc or count.value != expected_cores:
                raise ValueError('Unexpected target core count: '+label+': '+str(count.value))
            env[label] = count.value
        # Build identities, not only versions: preserve installed binary fingerprints.
        libs = set()
        for line in Path('/proc/self/maps').read_text().splitlines():
            path = line.split()[-1]
            if path.startswith('/') and '.so' in path and any(s in path for s in ('Ascend', 'ascend', 'torch_npu', 'tilelang', 'tvm', 'libcust')):
                libs.add(path)
        env['loaded_native_sha256'] = native_fingerprints(libs, a.task.resolve().parents[2])
        protocol = {**PROTOCOL, 'id': PROTOCOL_ID, 'core_sha256': hashlib.sha256((ROOT / 'tools/measure_baselines.py').read_bytes()).hexdigest()}
        identity = {'case': expected, 'library': library, 'sources': sources, 'environment': env,
                    'protocol': protocol, 'verifier_sha256': check['verifier_sha256'],
                    'wrapper_sha256': WRAPPER_SHA256, 'exported_flush': result['exported_flush'],
                    'adapter_sha256': check['adapter_sha256'], 'registry_sha256': check['providers_sha256']}
        # Library-specific binaries belong in the cache key; shared runtime
        # versions define which different libraries can be compared together.
        cohort = {k: env.get(k) for k in ('hostname', 'platform', 'torch', 'torch_npu', 'device_name', 'cann_home', 'ascend_npu_arch', 'soc_version', 'aicore_count', 'vectorcore_count')}
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
