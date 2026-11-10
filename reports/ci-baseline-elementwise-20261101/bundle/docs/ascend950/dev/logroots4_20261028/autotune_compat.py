"""Worker configuration search using TileLang AutoTuner, separate from formal M4.

Importing this helper alone uses the standard library. Call run_search only inside
the existing host npu-guard reservation, including all project imports in its caller.
"""

from __future__ import annotations

import datetime
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def run_search(*, factory, configs, compile_fn, supply_prog, reference_prog,
               output_dir, source_paths, control_config, warmup=3, repeats=15,
               rtol=1e-3, atol=1e-3, device=0, search_backend='event',
               profile_warmup=1, profile_repeats=3):
    """Search one fixed workload; return all trials and the fastest two configs.

    factory exposes the candidate names in its signature and returns a PrimFunc.
    compile_fn(**config) returns the existing builder's JITKernel. A factory can
    reuse that builder's .prim_func for elaboration. Both should read workload
    constants from module globals, avoiding non-scalar factory closure captures.
    supply_prog(params) supplies only input arguments on the NPU. The existing
    complete M3/boundary scripts remain required for selected production sources.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not configs or control_config not in configs:
        raise ValueError('configs must include the untuned production control')
    if search_backend not in ('event', 'profile_short'):
        raise ValueError('search_backend must be event or profile_short')
    if not isinstance(profile_warmup, int) or profile_warmup < 0 or not isinstance(profile_repeats, int) or profile_repeats < 1:
        raise ValueError('short-profile counts must be valid nonnegative/positive integers')
    if not isinstance(warmup, int) or warmup < 1 or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('warmup/repeats must be positive integers')
    config_keys = [json.dumps(config, sort_keys=True, allow_nan=False) for config in configs]
    if len(set(config_keys)) != len(configs):
        raise ValueError('duplicate configurations are not independent trials')

    # Pin compilation concurrency and disable cross-run tuning-score reuse.
    # The guard limits the entire process; timeout=0 below avoids detached timed-out
    # benchmark subthreads continuing while another candidate starts.
    os.environ['TILELANG_AUTO_TUNING_CPU_COUNTS'] = '1'
    os.environ['TILELANG_AUTO_TUNING_MAX_CPU_COUNT'] = '1'
    os.environ['TILELANG_AUTO_TUNING_DISABLE_CACHE'] = '1'
    import torch
    import torch_npu  # noqa: F401
    from tilelang.autotuner import AutoTuner
    import tilelang.autotuner.tuner as tuner_module

    import inspect
    legacy = "on_trial" not in inspect.signature(AutoTuner.run).parameters
    kernel_configs = {}
    short_timer = None
    timer_sources = set()
    if search_backend == 'profile_short':
        root = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(root))
        from tools.measure_baselines import _time_reference
        short_timer = _time_reference
        timer_sources.update({root / 'tools/measure_baselines.py',
                              root / 'tools/baseline_protocol.py'})

    torch.npu.set_device(device)
    paths = sorted({Path(p).resolve() for p in source_paths} | {
        Path(__file__).resolve(), Path(tuner_module.__file__).resolve()} | timer_sources)
    before = {str(p): _sha(p) for p in paths}
    (output / 'sources').mkdir()
    for index, path in enumerate(paths):
        shutil.copy2(path, output / 'sources' / f'{index:02d}_{path.name}')
    _write_json(output / 'sources.json', before)
    _write_json(output / 'request.json', {
        'configs': configs, 'control_config': control_config, 'warmup': warmup,
        'repeats': repeats, 'rtol': rtol, 'atol': atol,
        'purpose': 'configuration_search_only_not_formal_M4',
        'backend': search_backend, 'device': f'npu:{device}',
        'profile_warmup': profile_warmup, 'profile_repeats': profile_repeats,
        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})

    def strict_check(actual, reference):
        if len(actual) != len(reference):
            raise AssertionError('output count mismatch')
        for got, ref in zip(actual, reference):
            if got.shape != ref.shape or got.dtype != ref.dtype or got.device != ref.device:
                raise AssertionError('output shape/dtype/device mismatch')
            if got.is_floating_point():
                if not torch.equal(torch.isnan(got), torch.isnan(ref)):
                    raise AssertionError('NaN positions differ')
                inf = torch.isinf(ref)
                if not torch.equal(torch.isinf(got), inf):
                    raise AssertionError('Inf positions differ')
                if not torch.equal(torch.signbit(got[inf]), torch.signbit(ref[inf])):
                    raise AssertionError('Inf signs differ')
                zero = ref == 0
                if not bool((got[zero] == 0).all()) or not torch.equal(
                        torch.signbit(got[zero]), torch.signbit(ref[zero])):
                    raise AssertionError('reference-zero values or signs differ')
                finite = torch.isfinite(ref)
                if not bool(torch.isclose(got[finite], ref[finite], rtol=rtol, atol=atol).all()):
                    raise AssertionError('finite values exceed original tolerance')
            elif not torch.equal(got, ref):
                raise AssertionError('non-floating values differ')
            for supplied, _snapshot in getattr(validation_context, 'groups', []):
                for value in supplied:
                    if isinstance(value, torch.Tensor) and got.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
                        raise AssertionError('kernel output aliases an input')

    validation_context = threading.local()
    input_checks = []
    profile_outcomes = deque()

    def fresh_supply(params):
        supplied = supply_prog(params)
        if not isinstance(supplied, (list, tuple)):
            raise TypeError('supply_prog must return input arguments as a list/tuple')
        fresh = [value.clone() if isinstance(value, torch.Tensor) else value for value in supplied]
        snapshot = [value.clone() if isinstance(value, torch.Tensor) else value for value in fresh]
        validation_context.groups.append((fresh, snapshot))
        return fresh

    class NpuAutoTuner(AutoTuner):
        def _benchmark_target(self, *args, **kwargs):
            # Ascend's device context is thread-local; the stock tuner only binds
            # CUDA devices in its worker. Keep the actual search implementation.
            torch.npu.set_device(device)
            validation_context.groups = []
            profile_record = None
            try:
                if short_timer is None:
                    return super()._benchmark_target(*args, **kwargs)
                # Profile mode must never call the stock event-based benchmark.
                # Keep AutoTuner's supplied-input and manual correctness contracts,
                # then measure only through the isolated msprof/FFTS protocol.
                kernel = kwargs['jit_kernel']
                state = kwargs['benchmark_state']
                profiler = kernel.get_profiler(tensor_supply_type=self.profile_args.supply_type)
                params = profiler._get_params(with_output=False)
                inputs = tuple(fresh_supply(params))
                # This remote release's manual_assert_close synchronizes CUDA.
                # Keep the strict output/input contract and synchronize the NPU
                # explicitly without invoking that CUDA-only convenience API.
                actual = kernel(*inputs)
                expected = reference_prog(*inputs)
                torch.npu.synchronize()
                strict_check(actual if isinstance(actual, (tuple, list)) else [actual],
                             expected if isinstance(expected, (tuple, list)) else [expected])
                state.jit_input_tensors = list(inputs)
                state.jit_input_params = params
                profile = short_timer(torch, torch_npu, kernel, inputs,
                                      profile_warmup, profile_repeats)
                profile_record = {'warmup': profile_warmup, 'repeats': profile_repeats,
                                  'event_used': False, 'formal_M4_eligible': False, **profile}
                return profile['latency_ms'], None
            finally:
                torch.npu.synchronize()
                unchanged = True
                for values, snapshots in validation_context.groups:
                    for value, snapshot in zip(values, snapshots):
                        if isinstance(value, torch.Tensor):
                            unchanged = unchanged and torch.equal(
                                value.contiguous().view(torch.uint8),
                                snapshot.contiguous().view(torch.uint8))
                input_checks.append({'input_unchanged': bool(unchanged),
                                     'supplied_groups': len(validation_context.groups)})
                if search_backend == 'profile_short':
                    # One serial benchmark worker; outcomes are consumed in the
                    # same FIFO order as benchmark results, excluding compile errors.
                    if legacy:
                        config = kernel_configs[id(kwargs['jit_kernel'])]
                        on_trial({'index': configs.index(config), 'config': config,
                                  'status': 'ok' if profile_record is not None else 'benchmark_error',
                                  'validation': 'passed' if profile_record is not None else 'failed',
                                  'latency_ms': profile_record['latency_ms'] if profile_record else None,
                                  'search_profile': profile_record})
                    else:
                        profile_outcomes.append(profile_record)
                validation_context.groups = []
                if not unchanged:
                    raise AssertionError('candidate or reference mutated supplied inputs')

    def compile_candidate(**config):
        torch.npu.set_device(device)
        config.pop('__pass_configs__', None)
        try:
            kernel = compile_fn(**config)
        except Exception as exc:
            if legacy and not any(t['config'] == config for t in trials):
                on_trial({'index': configs.index(config), 'config': config, 'status': 'compile_error', 'error': repr(exc)})
            raise
        kernel_configs[id(kernel)] = config
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        (output / f'candidate_{key}.c').write_text(kernel.get_kernel_source())
        return kernel

    trials = []
    reporting_errors = []

    def on_trial(record):
        # TrialReporter otherwise disables a failed sink and continues silently.
        try:
            if 'search_profile' not in record and search_backend == 'profile_short' and record['status'] in ('ok', 'benchmark_error', 'timeout'):
                if not profile_outcomes:
                    raise RuntimeError('missing short-profile outcome for benchmark result')
                record['search_profile'] = profile_outcomes.popleft()
                if record['status'] == 'ok' and record['search_profile'] is None:
                    raise RuntimeError('successful trial has no short-profile samples')
            trials.append(record)
            _write_json(output / 'trials.json', trials)
            print('AUTOTUNER_TRIAL ' + json.dumps(record, allow_nan=False), flush=True)
        except Exception as error:
            reporting_errors.append(repr(error))

    tuner = NpuAutoTuner.from_kernel(factory, configs=configs)
    tuner.set_compile_args(target='ascend', out_idx=[-1], execution_backend='auto')
    tuner.set_profile_args(supply_prog=fresh_supply, ref_prog=reference_prog,
                           manual_check_prog=strict_check, rtol=rtol, atol=atol,
                           max_mismatched_ratio=0, skip_check=False,
                           cache_input_tensors=False, backend='msprof' if short_timer else 'event')
    tuner.jit_compile = compile_candidate
    tuner.jit_elaborate = lambda **cfg: compile_candidate(**cfg).prim_func
    previous_cwd = Path.cwd()
    failure = None
    artifact = None
    try:
        # Stock logger opens autotuner.log in cwd; keep it in this attempt.
        os.chdir(output)
        artifact = tuner.run(warmup=warmup, rep=repeats, timeout=0,
                             use_pipeline=False, enable_grouped_compile=False,
                             benchmark_multi_gpu=False, early_stop=False,
                             **({} if legacy else {'on_trial': on_trial}))
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        os.chdir(previous_cwd)
        torch.npu.synchronize()
        after = {str(p): _sha(p) for p in paths}
        changed = [name for name in before if before[name] != after[name]]
        _write_json(output / 'status.json', {
            'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'error': failure, 'source_changed': changed,
            'observed_trials': len(trials), 'expected_trials': len(configs),
            'reporting_errors': reporting_errors,
            'completed_search': False, 'input_checks': input_checks,
            'search_backend': search_backend,
            'purpose': 'configuration_search_only_not_formal_M4'})
    if changed or reporting_errors or len(trials) != len(configs):
        raise RuntimeError('source/reporting/trial-count integrity gate failed')
    if {r['index'] for r in trials} != set(range(len(configs))):
        raise RuntimeError('missing or duplicate terminal trial indices')
    successful = [r for r in trials if r['status'] == 'ok']
    if any(r['validation'] != 'passed' or not math.isfinite(r['latency_ms'])
           or r['latency_ms'] <= 0 for r in successful):
        raise RuntimeError('successful trial is unvalidated or has invalid timing')
    control = [r for r in successful if r['config'] == control_config]
    if len(control) != 1:
        raise RuntimeError('untuned production control failed search validation')
    ranked = sorted(successful, key=lambda r: r['latency_ms'])
    if not ranked or artifact.config != ranked[0]['config']:
        raise RuntimeError('AutoTuner result does not match recorded best trial')
    result = {'purpose': 'configuration_search_only_not_formal_M4',
              'search_backend': search_backend,
              'best_config': artifact.config, 'search_latency_ms': artifact.latency,
              'top_configs': [r['config'] for r in ranked[:2]],
              'control_config': control_config, 'control_search_latency_ms': control[0]['latency_ms'],
              'trials': trials, 'source_sha256': before,
              'formal_comparison': 'Use own M4/v2 for top candidates and control; do not merge search scores.'}
    _write_json(output / 'result.json', result)
    status = json.loads((output / 'status.json').read_text())
    status.update(completed_search=True, result_file=str(output / 'result.json'))
    _write_json(output / 'status.json', status)
    return result
