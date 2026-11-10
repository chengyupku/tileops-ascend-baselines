#!/usr/bin/env python3
"""Opt-in correctness phases; the existing ascend_pair_check.py is unchanged.

CPU static plan -> guarded original input generation/capture -> CPU oracle/compare.
Use ascend_pair_cpu.py for CPU commands. Device execution is only an entry for
the guarded subprocess, never a context kept alive across phase boundaries.
No profiling: use the original checker for profiling and non-whitelisted cases.
"""
import argparse
from dataclasses import asdict, replace
from contextlib import contextmanager
import hashlib
import inspect
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from ascend_phase_artifacts import (
    assert_preserved, canonical, contract, fingerprint, load_packet, packet,
    read_receipt, replace_storage_in_place, save_packet, seal, sha_file, unpack, write_new,
    dependency_closure, source_path, execution_fence, validate_admission, decode_admission, selected_admission,
)

# Explicit op/library pairs, independently audited against the original CPU
# reference switch. Adding a pair requires CPU and real-device review.
ALLOWLIST = {
    "AdaptiveMaxPool2dFwdOp": ("ops-nn", "ops-tensor"),
    "AdaptiveMaxPool2dIndicesFwdOp": ("ops-nn",),
    "AddFwdOp": ("ops-math", "ops-nn", "ops-tensor"),
    "AmaxFwdOp": ("ops-math", "ops-nn"),
}


# Original 25 manifest cases; original 42 pairs plus 16 authorized ops-tensor pairs.
CASE_ALLOWLIST = {
    "AdaptiveMaxPool2dFwdOp": ("global-1x1-float16", "spp-6x6-float16", "nondiv-7x7-bfloat16", "global-1x1-unbatched-float16"),
    "AdaptiveMaxPool2dIndicesFwdOp": ("global-1x1-float16", "spp-6x6-float16", "nondiv-7x7-bfloat16", "global-1x1-unbatched-float16"),
    "AddFwdOp": tuple(label + "-" + dtype for label in
                     ("hidden-state-prefill", "cnn-feat-broadcast", "llama-7b-ffn", "linear-bias")
                     for dtype in ("float16", "bfloat16", "float32")),
    "AmaxFwdOp": ("hidden-state-float16", "hidden-state-bfloat16", "long-seq-bfloat16", "3d-multidim-float16", "mlp-intermediate-float32"),
}


def configure(root):
    root = Path(root).resolve()
    sys.path[:0] = [str(root), str(root / "src"), str(root / "scripts"), str(root.parent)]
    return root


def permitted(op, library):
    if library not in ALLOWLIST.get(op, ()):
        raise ValueError(f"not in CPU phase allowlist: {op}/{library}; use original checker")


def case_for(op, case_id):
    if case_id not in CASE_ALLOWLIST.get(op, ()):
        raise ValueError("case not in original manifest whitelist; use original checker")
    from benchmarks.api import cases
    selected = [c for c in cases(op) if c.id == case_id]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one original manifest case, got {len(selected)}")
    return selected[0]


def case_snapshot(case):
    return dict(op=case._call.signature.name, case_id=case.id, parameters=case.params,
                manifest_params=case._call.params, indices=case._call.indices,
                declared_tensors=case._call.tensors,
                specs={k: None if v is None else asdict(v) for k, v in case._call.specs.items()},
                declared_input_names=list(case._call.signature.inputs),
                declared_output_names=list(case._call.signature.outputs),
                count_copies=case.count_copies)


def evidence_metadata(evidence):
    from workloads.numerics import Exact
    # Custom validators/probes may capture device tensors or call the candidate.
    # Partial would violate this checker's all-output requirement.
    if type(evidence) is not Exact:
        raise ValueError(f"{type(evidence).__name__} requires original checker; no CPU assumption")
    return dict(kind=evidence.kind, atol=evidence.atol, rtol=evidence.rtol,
                normalized=evidence.normalized,
                controls=[c.name for c in evidence.controls])


def sources(root, op, library, case_id=None):
    return dependency_closure(root, op, library, case_id)["sha256"]


def check_sources(root, expected, op, library, case_id=None):
    actual = sources(root, op, library, case_id)
    if actual != expected:
        changed = sorted(k for k in actual.keys() | expected.keys()
                         if actual.get(k) != expected.get(k))
        raise ValueError("dependency source fence changed: " + ", ".join(changed))


def check_plan_sources(root, meta):
    route = meta['dependency_closure']['selected_factory'].get('dynamic_route', {})
    selected = route.get('admission', {}).get('selected')
    check_sources(root, meta["source_sha256"], meta["op"], meta["library"], meta['case_id'] if selected else None)
    return check_plan_admission(root, meta)


def check_plan_admission(root, meta):
    route = meta['dependency_closure']['selected_factory'].get('dynamic_route', {})
    if 'admission' not in route:
        return None
    if route['admission']['selected']['case_id'] != meta['case_id'] or route['op'] != meta['op']:
        raise ValueError('selected admission case identity mismatch')
    return validate_admission(root, route)[0]


def check_source_bytes(root, expected):
    """Small startup fence; AST/dependency discovery stays outside the lock."""
    for key, digest in expected.items():
        path = source_path(root, key)
        actual = sha_file(path) if path.is_file() else None
        if actual != digest:
            raise ValueError("dependency source fence changed before capture: " + key)


def workload_state(case):
    """Only the audited metadata-only workload classes; no private device state."""
    from workloads.elementwise import ElementwiseCall
    from workloads.pool import AdaptiveMaxPool2dCall
    from workloads.reduction import ReductionCall
    wanted = {"AddFwdOp": ElementwiseCall, "AmaxFwdOp": ReductionCall,
              "AdaptiveMaxPool2dFwdOp": AdaptiveMaxPool2dCall,
              "AdaptiveMaxPool2dIndicesFwdOp": AdaptiveMaxPool2dCall}
    workload = case.workload
    if type(workload) is not wanted[case._call.signature.name]:
        raise ValueError("unreviewed workload class/private state requires original checker")
    allowed = {"ElementwiseCall": {"call", "_device", "name", "shape", "dtype"},
               "ReductionCall": {"call", "_device", "shape", "dtype"},
               "AdaptiveMaxPool2dCall": {"call", "_device", "output_size", "return_indices"}}
    if set(vars(workload)) != allowed[type(workload).__name__] or workload.call is not case._call:
        raise ValueError("complex/private workload attributes require original checker")
    state = packet({k: v for k, v in vars(workload).items() if k != "call"})
    if state["tensors"]:
        raise ValueError("workload constructor allocated tensors; requires original checker")
    return state["graph"]



def assert_cpu(value):
    from workloads.numerics import _tensors
    if any(t.device.type != "cpu" for t in _tensors(value)):
        raise ValueError("CPU phase received device tensors")


def original_reference(case, records):
    """Run the actual oracle and original verify/controls, with no candidate stub."""
    import torch
    from ascend_pair_check import observe_call
    from workloads.numerics import NegativeControl, verify
    evidence = case.verification
    evidence_metadata(evidence)
    expected = []
    controls = []
    observed = observe_call(torch, case.reference, records)

    def reference(*inputs):
        # verify() restores inputs before returning; inspect each oracle call
        # before restoration can hide writes, including NaN/zero bit changes.
        pristine = packet(inputs)
        try:
            result = observed(*inputs)
        finally:
            assert_preserved(pristine, packet(inputs))
        if not expected:
            expected.append(packet(result))
        return result

    def observe_control(control):
        def run(ref, args):
            controls.append(control.name)
            return control.run(ref, args)
        return NegativeControl(control.name, run)

    # Empty requests deliberately means NO candidate was called in this phase.
    verify(reference, case.inputs,
           evidence=replace(evidence, controls=tuple(observe_control(c) for c in evidence.controls)),
           requests={})
    if len(expected) != 1 or controls != [c.name for c in evidence.controls]:
        raise ValueError("reference OOM/incomplete controls cannot prepare a verified case")
    return expected[0], controls


def add_edge_inputs(inputs):
    """Original add_edge_check patterns, reconstructed from capture in CPU compare."""
    import torch
    x, y = unpack(packet(inputs))
    values_x = [-0., 0., 1., -1., float("inf"), -float("inf"), float("nan"),
                torch.finfo(x.dtype).tiny, torch.finfo(x.dtype).max]
    values_y = [-0., -0., -1., 1., -float("inf"), float("inf"), 1.,
                -torch.finfo(y.dtype).tiny, torch.finfo(y.dtype).max]
    x.view(-1)[:len(values_x)].copy_(torch.tensor(values_x, dtype=x.dtype, device="cpu"))
    y.view(-1)[:len(values_y)].copy_(torch.tensor(values_y, dtype=y.dtype, device="cpu"))
    return x, y


def prepare(root, op, case_id, library, output):
    """Static CPU plan only: no input draws, reference, verify, or candidate build."""
    import torch
    permitted(op, library)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    closure = dependency_closure(root, op, library, case_id)
    check_dynamic_admission(case_id, closure["selected_factory"])
    initial_sources = closure["sha256"]
    case = case_for(op, case_id)
    if any(s is not None and s.cpu for s in case._call.specs.values()):
        raise ValueError("mixed CPU/device manifest tensors require the original checker")
    state = workload_state(case)
    # This reads constructor metadata, not workload.arguments() (which for some
    # workloads materializes another random input). Tensor constructors fail.
    arguments = packet(case._call.arguments({}))
    if arguments["tensors"]:
        raise ValueError("construction-time tensor arguments require original checker")
    save_packet(output / "arguments.pt", arguments)
    seed = int.from_bytes(hashlib.sha256(f"{op}::{case_id}".encode()).digest()[:4], "big")
    check_sources(root, initial_sources, op, library, case_id)
    route = closure["selected_factory"].get("dynamic_route", {})
    if "admission" in route: validate_admission(root, route)
    meta = dict(plan_version=2, op=op, case_id=case_id, library=library,
                case_snapshot=case_snapshot(case), source_sha256=initial_sources,
                dependency_closure=closure, workload_state=state,
                input_seed=seed, input_generation_device="npu", reference_device="cpu",
                random_payload_identity="original factory and seed on NPU; capture bytes, never redraw in compare",
                manifest_value_seed_base=importlib.import_module("tileops.manifest.primitives").WORKLOAD_SEED,
                manifest_value_seed_rule="WORKLOAD_SEED xor crc32(op + ':' + tensor); original specs unchanged",
                candidate_calls=0, reference_calls=0, input_draws=0,
                verification_policy="Exact only; original declaration evaluated on captured original inputs",
                torch_version=torch.__version__, threads=torch.get_num_threads(),
                prepare_wall_seconds=time.monotonic() - start)
    write_new(output / "dependency_closure.json", canonical(closure) + b"\n")
    return seal(output, "prepare", json.loads(canonical(meta)))


def apply_add_edges(inputs):
    """Original add_edge_check's nine values, on the SAME factory input tensors."""
    import torch
    x, y = inputs
    values_x = [-0., 0., 1., -1., float("inf"), -float("inf"), float("nan"),
                torch.finfo(x.dtype).tiny, torch.finfo(x.dtype).max]
    values_y = [-0., -0., -1., 1., -float("inf"), float("inf"), 1.,
                -torch.finfo(y.dtype).tiny, torch.finfo(y.dtype).max]
    x.view(-1)[:len(values_x)].copy_(torch.tensor(values_x, dtype=x.dtype, device=x.device))
    y.view(-1)[:len(values_y)].copy_(torch.tensor(values_y, dtype=y.dtype, device=y.device))



def check_dynamic_admission(case_id, expected):
    route = expected.get('dynamic_route')
    if route and 'admission' in route and route['admission']['selected']['case_id'] != case_id:
        raise ValueError('selected admission case identity mismatch')
    if route and route['registered']:
        if case_id not in route['accepted_ids']:
            raise ValueError('case not accepted by original dynamic provider')
        if case_id in route.get('continuation_ids', []):
            raise ValueError('continuation runtime external receipt closure requires independent review')
        if case_id in route['pinned_ids']:
            raise ValueError('pinned runtime external receipt closure requires independent review')
        if route.get('hold'):
            raise ValueError(route['hold'])


def validate_dynamic_factory(root, factory, case, expected, loaded_admission=None):
    """Bind live resolver, partials/closure cells, JSON state and final dispatch.

    Resolves the original callable but never calls a provider, input generator,
    compiler or device API. Compiling source here extracts code objects only;
    no exec/eval is used to discover registrations.
    """
    from functools import partial
    from types import CodeType
    check_dynamic_admission(case.id, expected)
    route = expected['dynamic_route']
    package, op = route['package'], route['op']

    def function(fn, module, qualname):
        if not inspect.isfunction(fn) or fn.__module__ != module or fn.__qualname__ != qualname:
            raise ValueError('dynamic function identity mismatch: ' + module + ':' + qualname)
        mod = sys.modules.get(module)
        path = Path(inspect.getsourcefile(fn)).resolve()
        wanted = root / (module.replace('.', '/') + '.py')
        if not wanted.is_file(): wanted = root / module.replace('.', '/') / '__init__.py'
        if mod is None or fn.__globals__ is not vars(mod) or path != wanted.resolve():
            raise ValueError('dynamic function globals/source mismatch')
        code = compile(wanted.read_bytes(), str(wanted.resolve()), 'exec')
        def codes(node):
            yield node
            for value in node.co_consts:
                if isinstance(value, CodeType): yield from codes(value)
        matches = [node for node in codes(code) if node.co_qualname == qualname]
        if len(matches) != 1 or fn.__code__ != matches[0] or fn.__defaults__ or fn.__kwdefaults__:
            raise ValueError('dynamic function code/defaults mismatch: ' + qualname)
        return fn

    module = importlib.import_module(package)
    admission = None
    if 'admission' in route:
        if case._call.signature.name != op or case.id != route['admission']['selected']['case_id']:
            raise ValueError('selected admission case identity mismatch')
        admission, data = validate_admission(root, route)
        loaded_admission = admission if loaded_admission is None else loaded_admission
        loaded_data = decode_admission(package, route['continuation_registry'], loaded_admission['history'])
        if selected_admission(package, op, case.id, loaded_data) != admission['selected']:
            raise ValueError('loaded admission selected identity differs')
        data = loaded_data
        # Full loaded state MUST equal these actual on-disk bytes. The returned
        # evidence records those bytes; historical bytes are never labeled live.
        # Refresh expected full closure sets only after selected identity matches.
        route = dict(route, data_sha256={k: hashlib.sha256(canonical(v)).hexdigest() for k, v in data.items()},
                     accepted_ids=data['_enabled_cases' if package.endswith('.ops_math') else '_accepted'].get(op, []),
                     pinned_ids=data.get('_pinned', {}).get(op, []),
                     continuation_ids=data.get('_continuation', {}).get(op, []))
    for name, digest in route['data_sha256'].items():
        value = getattr(module, name)
        if name == '_continuation':
            if type(value) is not dict or any(type(v) is not set for v in value.values()):
                raise ValueError('loaded continuation table has unexpected types')
            value = {k: sorted(v) for k, v in value.items()}
        if hashlib.sha256(canonical(value)).hexdigest() != digest:
            raise ValueError('loaded dynamic admission JSON differs from plan: ' + name)
    registry = importlib.import_module('benchmarks.ascend_all_pairs.registry')
    function(registry.candidate_factory, registry.__name__, 'candidate_factory')
    case_library = 'ops-math' if package.endswith('.ops_math') else 'ops-nn'
    if registry.LIBRARY_MODULES.get(case_library) != package.rsplit('.', 1)[-1]:
        raise ValueError('loaded registry library mapping differs')
    providers = module.PROVIDERS
    if type(providers) is not dict or (op in providers) != route['registered']:
        raise ValueError('loaded dynamic provider route differs from plan')
    if not route['registered']:
        if type(factory) is not partial or factory.args != (case,) or factory.keywords:
            raise ValueError('original fallback partial binding differs')
        classic = importlib.import_module('benchmarks.ops.bench_ascend_classics')
        function(classic.candidate_factories, classic.__name__, 'candidate_factories')
        if classic.PROVIDERS.get(op, {}).get(case_library) is not factory.func:
            raise ValueError('original fallback registry identity differs')
        fn = factory.func
    else:
        function(factory, registry.__name__, 'candidate_factory.<locals>.<lambda>')
        cells = inspect.getclosurevars(factory).nonlocals
        if set(cells) != {'case', 'op', 'providers'} or cells['case'] is not case or cells['op'] != op or cells['providers'] is not providers:
            raise ValueError('original registry closure differs')
        fn = providers[op]
        if package.endswith('.ops_math'):
            if type(fn) is not partial or fn.args or fn.keywords != {'op': op} or fn.func is not module._public:
                raise ValueError('original public partial binding differs')
            function(fn.func, package, '_public')
            screened = module.CANDIDATES.get(op)
            if type(screened) is not partial or screened.args or screened.keywords != {'op': op, 'factory': module._add} or screened.func is not module._screened:
                raise ValueError('original screened partial binding differs')
            for name in ('_screened', '_add'):
                function(getattr(module, name), package, name)
            if module.NATIVE_CANDIDATES.get(op) is not module._add:
                raise ValueError('original native factory identity differs')
            semantic = importlib.import_module(package + '.semantic_limits')
            if module._semantic_reason is not semantic.reason:
                raise ValueError('original semantic filter identity differs')
            function(semantic.reason, semantic.__name__, 'reason')
            terminal = importlib.import_module('benchmarks.ascend_math_baselines')
            function(terminal.opsmath_add, terminal.__name__, 'opsmath_add')
            fn = fn.func
        else:
            function(fn, package, '_provider.<locals>.factory')
            cells = inspect.getclosurevars(fn).nonlocals
            keys = {'accepted_ids', 'name', 'pinned_ids'}
            if route.get('continuation_registry'):
                keys.add('continuation_ids')
                if type(cells.get('continuation_ids')) is not frozenset or cells['continuation_ids'] != frozenset(route['continuation_ids']):
                    raise ValueError('original continuation factory closure differs')
            if set(cells) != keys or cells['name'] != op or type(cells['accepted_ids']) is not frozenset or type(cells['pinned_ids']) is not frozenset or cells['accepted_ids'] != frozenset(route['accepted_ids']) or cells['pinned_ids'] != frozenset(route['pinned_ids']):
                raise ValueError('original accepted factory closure differs')
            staging = importlib.import_module(package + '.staging')
            matrix = importlib.import_module(package + '.matrix')
            catalog = importlib.import_module(package + '.catalog')
            if staging.CANDIDATES.get(op) is not matrix.legacy or catalog.LEGACY.get(op) != ('benchmarks.ascend_nn_direct', 'opsnn_inplace_add', 'norm/inplace_add_rms_norm'):
                raise ValueError('original NN dispatch target differs')
            function(matrix.legacy, matrix.__name__, 'legacy')
            terminal = importlib.import_module('benchmarks.ascend_nn_direct')
            function(terminal.opsnn_inplace_add, terminal.__name__, 'opsnn_inplace_add')
    function(fn, expected['module'], expected['qualname'])
    if Path(inspect.getsourcefile(fn)).resolve() != source_path(root, expected['source']).resolve():
        raise ValueError('actual dynamic provider source differs from plan')
    return {'module': fn.__module__, 'qualname': fn.__qualname__,
            'dynamic_binding': 'original resolver/code/closure/JSON/terminal factory checked',
            **({'admission_current': admission, 'admission_loaded': loaded_admission} if admission is not None else {})}


def validate_selected_factory(root, factory, case, expected, loaded_admission=None):
    """Check the actual unmodified registry's selected callable without building."""
    if expected.get("dynamic_route"):
        return validate_dynamic_factory(root, factory, case, expected, loaded_admission)
    from functools import partial
    if isinstance(factory, partial):
        if factory.args != (case,) or factory.keywords:
            raise ValueError("unreviewed factory binding/private args")
        fn = factory.func
    elif inspect.isfunction(factory):
        closure = inspect.getclosurevars(factory).nonlocals
        if set(closure) != {"case", "op", "providers"} or closure["case"] is not case:
            raise ValueError("unreviewed factory closure/private state")
        if closure["op"] != case._call.signature.name or type(closure["providers"]) is not dict:
            raise ValueError("factory closure identity mismatch")
        fn = closure["providers"][closure["op"]]
        if Path(inspect.getsourcefile(factory)).resolve() != (root / "benchmarks/ascend_all_pairs/registry.py").resolve():
            raise ValueError("factory wrapper is not the original registry")
    else:
        raise ValueError("unreviewed dynamic factory callable")
    identity = {"module": getattr(fn, "__module__", None), "qualname": getattr(fn, "__qualname__", None)}
    if identity != {k: expected[k] for k in ("module", "qualname")}:
        raise ValueError("actual provider differs from static dependency closure")
    if Path(inspect.getsourcefile(fn)).resolve() != source_path(root, expected["source"]).resolve():
        raise ValueError("actual provider source differs from static dependency closure")
    return identity


def validate_timing_mode(meta, mode):
    # These original providers use host orchestration for packing and repeated
    # native launches. All of that remains guarded and captured; no performance
    # measurement or timing-mode relabeling occurs in this correctness path.
    if mode == "device_async":
        return
    if (mode == "host_roundtrip" and meta["library"] == "ops-tensor"
            and meta["op"] in ("AddFwdOp", "AdaptiveMaxPool2dFwdOp")):
        return
    raise ValueError("unreviewed host-roundtrip candidate requires original checker")


def finish_device(sync, candidate, capture_inputs):
    """Stop device work at the first cleanup/sync fault; let the child exit."""
    failures = []
    def attempt(stage, fn):
        try:
            fn()
        except Exception as exc:
            failures.append({"stage": stage, "error_type": type(exc).__name__, "error": str(exc)})
        return not failures
    if not attempt("pre_cleanup_sync", sync):
        return failures
    if candidate is not None and candidate.cleanup is not None:
        if not attempt("candidate_cleanup", candidate.cleanup):
            return failures
    if not attempt("post_cleanup_sync", sync):
        return failures
    if not attempt("capture_inputs_and_final_identity", capture_inputs):
        return failures
    attempt("final_sync", sync)
    return failures


def device_capture(root, prep_path, output, *, mock_factory=None):
    """Whole guarded foreground child; no CPU reference or numerical compare.

    The only unguarded entry is the explicit CPU test mock, never CLI/native.
    All device-capable imports and allocation stay after require_guard().
    """
    if mock_factory is None:
        from ascend_pair_check import require_guard
        require_guard()
    import torch
    from ascend_pair_check import observe_call
    from workloads.device import set_run_device
    from tools.baseline_protocol import seed_case
    from benchmarks.ascend_baselines import UnsupportedCase
    device = "cpu" if mock_factory is not None else "npu"
    if mock_factory is None:
        import torch_npu  # guarded whole-process boundary only
    output = Path(output)
    row = dict(status="failed", execution_kind="cpu_mock" if mock_factory else "npu",
               runtime_host=socket.gethostname(), pid=os.getpid(), process_group=os.getpgrp(),
               torch_version=torch.__version__, observed_calls={"candidate": []},
               started_at=datetime.now(timezone.utc).isoformat(), sync_complete=False,
               reference_calls=0, numerical_comparisons=0, native_acceptance="hold",
               native_hold_reason="independent native source/build/load/launch and NPU codec audit required")
    row["native_environment"] = {key: os.environ.get(key) for key in (
        "ASCEND_CUSTOM_OPP_PATH", "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH",
        "ASCEND_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TILELANG_CACHE_DIR", "BASELINE_RUN_ID")}
    candidate, case, source_context = None, None, None
    started, stage = time.monotonic(), "load_plan"
    stage_started = started
    row["stage_seconds"] = {}
    def mark_stage(name):
        nonlocal stage, stage_started
        now = time.monotonic()
        row["stage_seconds"][stage] = row["stage_seconds"].get(stage, 0) + now - stage_started
        stage, stage_started = name, now
    sync = (lambda: None) if mock_factory else torch.npu.synchronize
    try:
        prep = read_receipt(prep_path, "prepare")
        meta = prep["metadata"]
        if meta.get("plan_version") != 2:
            raise ValueError("requires a v2 static plan, not CPU-generated v1 inputs")
        permitted(meta["op"], meta["library"])
        check_source_bytes(root, meta["source_sha256"])
        route = meta['dependency_closure']['selected_factory'].get('dynamic_route', {})
        for directory in route.get('native_directories', []):
            expected = {key for key in meta['source_sha256'] if key.startswith(directory + '/')}
            actual = {str(path.resolve().relative_to(root.parent)) for path in (root.parent / directory).rglob('*')
                      if path.is_file() and '__pycache__' not in path.parts}
            if actual != expected:
                raise ValueError('original native inventory changed before capture: ' + directory)
        row["admission_before_capture"] = check_plan_admission(root, meta)
        row.update(op=meta["op"], case_id=meta["case_id"], library=meta["library"],
                   prepare_receipt=Path(prep_path).name)
        set_run_device(device)
        # Exactly the original seed_case and manifest factory, before first input
        # access. Never transplant a CPU RNG payload into the real device path.
        row["input_seed"] = seed_case(torch, meta["op"], meta["case_id"])
        if row["input_seed"] != meta["input_seed"]:
            raise ValueError("original seed derivation changed")
        case = case_for(meta["op"], meta["case_id"])
        if json.loads(canonical(case_snapshot(case))) != meta["case_snapshot"]:
            raise ValueError("manifest case changed before device build")
        if workload_state(case) != meta["workload_state"]:
            raise ValueError("workload private state changed before input generation")
        # Resolve without building. Its exact callable must match the static
        # selected-provider dependency roots. The resolver remains unchanged.
        if mock_factory is None:
            from benchmarks.ascend_all_pairs.registry import candidate_factory
            factory = candidate_factory(case, meta["library"])
            row["selected_factory"] = validate_selected_factory(root, factory, case, meta["dependency_closure"]["selected_factory"])
        else:
            factory = lambda: mock_factory(case)
        # Exercise the same executed-source boundary in CPU controls. Their
        # explicit mock callables live outside the project source roots, while
        # original case generation and all project helper calls stay fenced.
        source_context = execution_fence(root, meta["source_sha256"],
                                         row.setdefault("executed_python_sources", {}))
        source_context.__enter__()
        mark_stage("generate_original_inputs")
        original_inputs = case.inputs
        sync()
        original = packet(original_inputs)
        save_packet(output / "inputs_before_build.pt", original)
        row["input_generation_device"] = device
        row["input_factory"] = type(case.workload).__module__ + "." + type(case.workload).__qualname__
        row["verification_declaration"] = evidence_metadata(case.verification)
        mark_stage("build")
        row["tvm_module_before_build"] = getattr(sys.modules.get("tvm"), "__file__", None)
        if mock_factory is None:
            row['selected_factory_before_build'] = validate_selected_factory(root, factory, case, meta['dependency_closure']['selected_factory'], row['selected_factory'].get('admission_loaded'))
        candidate = factory()
        if meta['dependency_closure']['selected_factory'].get('dynamic_route', {}).get('admission'):
            for name, digest in candidate.source_sha256.items():
                if meta['source_sha256'].get(name) != digest:
                    raise ValueError('candidate source/vendor outside planned identity: ' + name)
        row.update(api=candidate.api, adaptation=candidate.adaptation,
                   source_sha256=dict(candidate.source_sha256), limitations=candidate.limitations,
                   timing_mode=candidate.timing_mode, count_copies=True)
        sync()
        save_packet(output / "inputs_after_build.pt", packet(case.inputs))
        # No injected constructor arguments: record what the original provider
        # accessed, or the same static constructor metadata when unused.
        arguments = packet(case.__dict__.get("arguments", unpack(load_packet(Path(prep_path).parent / "arguments.pt"))))
        if arguments["tensors"]:
            raise ValueError("provider materialized private constructor tensors")
        save_packet(output / "arguments.pt", arguments)
        if workload_state(case) != meta["workload_state"]:
            raise ValueError("provider changed workload private state")
        impl = candidate.implementation
        if impl.args is not None or impl.noncomparable_reason is not None:
            raise ValueError("private arguments/noncomparable implementations require original checker")
        validate_timing_mode(meta, candidate.timing_mode)
        run = observe_call(torch, impl.run, row["observed_calls"]["candidate"])
        mark_stage("execute")
        if impl.reset is not None:
            impl.reset()
        sync()
        save_packet(output / "inputs_before_call.pt", packet(case.inputs))
        with torch.no_grad():
            actual = run(*case.inputs)
        sync()
        # Joint captures also retain output/input aliases and repeated identity.
        save_packet(output / "joint_after_call.pt", packet((case.inputs, actual)))
        save_packet(output / "actual.pt", packet(actual))
        save_packet(output / "inputs_after_call.pt", packet(case.inputs))
        if meta["op"] == "AddFwdOp":
            mark_stage("edge_execute")
            apply_add_edges(case.inputs)
            sync()
            save_packet(output / "edge_before_call.pt", packet(case.inputs))
            with torch.no_grad():
                edge_actual = run(*case.inputs)
            sync()
            save_packet(output / "edge_joint_after_call.pt", packet((case.inputs, edge_actual)))
            save_packet(output / "edge_actual.pt", packet(edge_actual))
            save_packet(output / "edge_after_call.pt", packet(case.inputs))
            replace_storage_in_place(case.inputs, original)
            sync()
            save_packet(output / "inputs_restored.pt", packet(case.inputs))
        row["status"] = "executed_uncompared"
    except UnsupportedCase as exc:
        row.update(status="unsupported", stage=stage, error=str(exc))
    except Exception as exc:
        row.update(status="failed", stage=stage, error_type=type(exc).__name__,
                   error=str(exc), traceback=traceback.format_exc(), device_work_aborted=True)
    finally:
        mark_stage("cleanup_and_sync")
        def capture_final_state():
            if mock_factory is None and candidate is not None:
                row['selected_factory_after_cleanup'] = validate_selected_factory(root, factory, case, meta['dependency_closure']['selected_factory'], row['selected_factory'].get('admission_loaded'))
            if candidate is not None:
                final_hashes = dict(candidate.source_sha256)
                for path, digest in row.get("source_sha256", {}).items():
                    if final_hashes.get(path) != digest:
                        raise ValueError("candidate changed an earlier source identity: " + path)
                row["source_sha256"] = final_hashes
            if case is not None and "inputs" in case.__dict__:
                save_packet(output / "inputs_after_cleanup.pt", packet(case.inputs))
        # A fault may surface from a provider or synchronization call. Do not
        # issue another device operation on a possibly faulted context. Only
        # publish CPU metadata before this foreground child exits nonzero.
        if row.get("device_work_aborted"):
            failures = []
            row["cleanup_skipped_due_to_prior_fault"] = True
        else:
            failures = finish_device(sync, candidate, capture_final_state)
        row["cleanup_errors"] = failures
        row["sync_complete"] = not failures and not row.get("device_work_aborted", False)
        if failures:
            row.update(status="device_sync_or_cleanup_failed", sync_error=repr(failures))
        if source_context is not None:
            source_context.__exit__(None, None, None)
        mark_stage("publication")
        row.update(wall_seconds=time.monotonic() - started,
                   finished_at=datetime.now(timezone.utc).isoformat())
        write_new(output / "capture.json", canonical(row) + b"\n")
    return 0 if row["status"] in ("executed_uncompared", "unsupported") else 1



def validate_capture(prep_path, capture_path, *, allow_mock=False):
    prep = read_receipt(prep_path, "prepare")
    execution = read_receipt(capture_path, "execute")
    meta = execution["metadata"]
    row = json.loads((Path(capture_path).parent / "capture.json").read_bytes())
    if meta["prepare_receipt"] != Path(prep_path).name or row["prepare_receipt"] != Path(prep_path).name:
        raise ValueError("execution belongs to another prepared payload")
    if meta["guard_returncode"] != 0 or not meta["child_process_exited"]:
        raise ValueError("guard failed/not exited: invalidate all captured output")
    if prep["metadata"].get("plan_version") != 2:
        raise ValueError("requires a v2 static plan")
    if row["torch_version"] != prep["metadata"]["torch_version"]:
        raise ValueError("CPU and device Torch versions differ; use matching runtimes")
    if row["execution_kind"] != "npu" and not (allow_mock and row["execution_kind"] == "cpu_mock"):
        raise ValueError("CPU mock is not device execution evidence")
    if not row["sync_complete"] or row["status"] not in ("executed_uncompared", "unsupported"):
        raise ValueError("device stage did not complete successfully")
    wanted = (2 if prep["metadata"]["op"] == "AddFwdOp" else 1) if row["status"] != "unsupported" else 0
    calls = row["observed_calls"]["candidate"]
    if len(calls) != wanted or not all(c.get("returned") for c in calls):
        raise ValueError("missing actual candidate call evidence")
    for key in ("op", "case_id", "library"):
        if row[key] != prep["metadata"][key]:
            raise ValueError("candidate identity mismatch")
    return prep, row


def compare(root, prep_path, capture_path, output, *, allow_mock=False):
    """CPU only: original oracle/verify on captured NPU bytes; never draw/run candidate."""
    import torch
    from workloads.numerics import compare_outputs
    from workloads.device import set_run_device
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", candidate_calls_in_cpu_compare=0,
                  prepare_receipt=Path(prep_path).name, execution_receipt=Path(capture_path).name,
                  native_acceptance="hold", native_pairs_accepted=0,
                  native_hold_reason="independent native source/build/load/launch and real NPU codec review required")
    try:
        prep, row = validate_capture(prep_path, capture_path, allow_mock=allow_mock)
        meta = prep["metadata"]
        if torch.__version__ != meta["torch_version"]:
            raise ValueError("compare CPU and planned Torch versions differ; use matching runtimes")
        result["admission_current"] = check_plan_sources(root, meta)
        permitted(meta["op"], meta["library"])
        result.update({key: meta[key] for key in (
            "op", "case_id", "library", "case_snapshot", "input_seed",
            "reference_device", "random_payload_identity")})
        result["input_generation_device"] = row.get("input_generation_device", "not_generated")
        result["reference_engine"] = "original manifest case.reference and verify on CPU, captured original inputs"
        result["independent_cpu_reference"] = True
        if row["status"] == "unsupported":
            result.update(status="unsupported", reason=row["error"], stage=row["stage"])
            write_new(output / "result.json", canonical(result) + b"\n")
            return seal(output, "compare", {"status": result["status"]})
        for relative, digest in row["source_sha256"].items():
            if sha_file(root.parent / relative) != digest:
                raise ValueError("candidate native/JIT source or binary changed: " + relative)
        for relative, digest in row.get("executed_python_sources", {}).items():
            if meta["source_sha256"].get(relative) != digest or sha_file(source_path(root, relative)) != digest:
                raise ValueError("executed dependency source fence mismatch: " + relative)
        set_run_device("cpu")
        case = case_for(meta["op"], meta["case_id"])
        captured = Path(capture_path).parent
        wanted_device = "cpu" if row["execution_kind"] == "cpu_mock" else "npu"
        def captured_packet(name):
            p = load_packet(captured / name)
            if any(m["device"].split(":")[0] != wanted_device for m in p["tensors"]):
                raise AssertionError("candidate returned/read tensors on the wrong device")
            return p
        inputs = captured_packet("inputs_before_build.pt")
        if row["input_seed"] != meta["input_seed"] or row["input_generation_device"] != wanted_device:
            raise ValueError("original input generation provenance mismatch")
        case.__dict__["inputs"] = unpack(inputs)
        arguments = load_packet(captured / "arguments.pt")
        assert_preserved(load_packet(Path(prep_path).parent / "arguments.pt"), arguments)
        case.__dict__["arguments"] = unpack(arguments)
        if json.loads(canonical(case_snapshot(case))) != meta["case_snapshot"] or workload_state(case) != meta["workload_state"]:
            raise ValueError("manifest/workload parameters changed")
        declaration = evidence_metadata(case.verification)
        if declaration != row["verification_declaration"]:
            raise ValueError("verification policy changed")
        result["verification_declaration"] = declaration
        for name in ("inputs_after_build.pt", "inputs_before_call.pt",
                     "inputs_after_call.pt", "inputs_after_cleanup.pt"):
            assert_preserved(inputs, captured_packet(name))
        produced = captured_packet("actual.pt")
        joint = captured_packet("joint_after_call.pt")
        joint_inputs, joint_output = unpack(joint)
        assert_preserved(inputs, packet(joint_inputs))
        assert_preserved(produced, packet(joint_output))
        records, executed = [], {}
        with execution_fence(root, meta["source_sha256"], executed):
            expected, controls = original_reference(case, records)
        assert_preserved(inputs, packet(case.inputs))
        save_packet(output / "expected.pt", expected)
        check = compare_outputs(unpack(produced), unpack(expected), case.verification)
        if not check.checked_outputs or check.checked_outputs != check.total_outputs:
            raise ValueError("candidate did not verify every output")
        if meta["op"] == "AddFwdOp":
            # Derive only the deterministic boundary edits from captured original
            # inputs. This is not a random draw and never invokes the candidate.
            edge = packet(add_edge_inputs(unpack(inputs)))
            for name in ("edge_before_call.pt", "edge_after_call.pt"):
                assert_preserved(edge, captured_packet(name))
            assert_preserved(inputs, captured_packet("inputs_restored.pt"))
            edge_joint_inputs, edge_joint_output = unpack(captured_packet("edge_joint_after_call.pt"))
            assert_preserved(edge, packet(edge_joint_inputs))
            actual_edge = unpack(captured_packet("edge_actual.pt"))
            assert_preserved(packet(actual_edge), packet(edge_joint_output))
            edge_inputs = unpack(edge)
            from ascend_pair_check import observe_call
            pristine = packet(edge_inputs)
            with execution_fence(root, meta["source_sha256"], executed):
                try:
                    expected_edge = observe_call(torch, case.reference, records)(*edge_inputs)
                finally:
                    assert_preserved(pristine, packet(edge_inputs))
            save_packet(output / "edge_expected.pt", packet(expected_edge))
            torch.testing.assert_close(actual_edge, expected_edge, rtol=0, atol=0, equal_nan=True)
            zero = expected_edge == 0
            torch.testing.assert_close(torch.signbit(actual_edge)[zero], torch.signbit(expected_edge)[zero])
            result["edge_check"] = dict(changed_inputs=True, extremes=True, nonfinite=True,
                                        signed_zero=True, independent_cpu_reference=True)
        result.update(status="cpu_mock_verified" if row["execution_kind"] == "cpu_mock" else "numerically_verified",
                      verification=asdict(check), inputs_preserved=True,
                      input_fingerprint=fingerprint(inputs), output_fingerprint=fingerprint(produced),
                      joint_input_output_fingerprint=fingerprint(joint),
                      negative_controls_rejected=controls,
                      observed_calls={"reference": records, "candidate": row["observed_calls"]["candidate"]},
                      executed_reference_sources=executed,
                      api=row["api"], adaptation=row["adaptation"], source_sha256=row["source_sha256"],
                      limitations=row["limitations"], count_copies=row["count_copies"],
                      timing_mode=row["timing_mode"], profile_requested=False,
                      native_environment=row["native_environment"],
                      native_attribution="numerics only; source/call/log evidence requires independent native audit")
        result["admission_after_compare"] = check_plan_sources(root, meta)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    write_new(output / "result.json", canonical(result) + b"\n")
    return seal(output, "compare", {"status": result["status"], "native_acceptance": "hold"})



def guarded_run(root, args):
    """CPU-only parent; guard wraps ALL runtime initialization through exit."""
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    prep_path = prepare(root, args.op, args.case_id, args.library, destination / "prepare")
    prep = read_receipt(prep_path, "prepare")
    check_plan_sources(root, prep["metadata"])
    capture = destination / "execute"
    capture.mkdir()
    launch = json.loads(args.device_launch_json)
    if not isinstance(launch, list) or not launch or not all(isinstance(v, str) for v in launch):
        raise ValueError("device-launch-json must be an argv list ending in a Python executable")
    # Use the same host guard implementation and host lock; bypass only Python
    # site startup, not the guard. Never run a shell wrapper with unknown flags.
    guard_script = getattr(args, "guard_script", None) or root.parent / "tools/npu_guard.py"
    if not Path(guard_script).is_file():
        guard_script = Path("/root/dyq/workspace/tools/npu_guard.py")
    if not guard_script.is_file():
        raise ValueError("supply --guard-script pointing to the host's existing npu_guard.py")
    guard_command = [sys.executable, "-I", "-S", "-B", str(guard_script.resolve())]
    command = guard_command + ["run", "--owner", args.owner, "--wait", str(args.wait),
                               "--timeout", str(args.timeout), "--"] + launch + [
        str(Path(__file__).resolve()), "--root", str(root), "device",
        "--prepared", str(prep_path), "--output", str(capture)]
    started = time.monotonic()
    requested_at = datetime.now(timezone.utc).isoformat()
    with (capture / "guard.stdout.log").open("xb") as out, (capture / "guard.stderr.log").open("xb") as err:
        child = subprocess.run(command, stdout=out, stderr=err, check=False)
    guard_meta = dict(prepare_receipt=prep_path.name, guard_returncode=child.returncode,
                      child_process_exited=True, argv=command,
                      requested_at=requested_at, finished_at=datetime.now(timezone.utc).isoformat(),
                      guarded_wall_seconds=time.monotonic() - started,
                      native_evidence_sources=[str(p.resolve()) for p in args.native_evidence])
    write_new(capture / "guard.json", canonical(guard_meta) + b"\n")
    # Hash/copy logs and full output fingerprints ONLY after guard has returned.
    for i, source in enumerate(args.native_evidence):
        source = source.resolve()
        if source.is_symlink():
            raise ValueError("native evidence symlinks are forbidden")
        target = capture / "native_evidence" / str(i)
        target.parent.mkdir(exist_ok=True)
        if source.is_file():
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        elif source.is_dir():
            if any(p.is_symlink() for p in source.rglob("*")):
                raise ValueError("native evidence contains symlinks")
            shutil.copytree(source, target)
        else:
            raise ValueError("missing native evidence path")
    admission = check_plan_sources(root, prep["metadata"])
    write_new(capture / "admission_postguard.json", canonical(admission) + b"\n")
    capture_path = seal(capture, "execute", guard_meta)
    # 75/76/78/124 and every other nonzero code invalidate the run, even if a
    # candidate returned and left apparently correct tensors before interruption.
    if child.returncode:
        failed = destination / "compare"
        failed.mkdir()
        write_new(failed / "result.json", canonical(dict(
            status="invalid_guard_run", guard_returncode=child.returncode,
            execution_receipt=capture_path.name, candidate_calls_in_cpu_compare=0)) + b"\n")
        print(seal(failed, "compare", {"status": "invalid_guard_run"}), flush=True)
        return child.returncode
    result = compare(root, prep_path, capture_path, destination / "compare")
    print(result, flush=True)
    return 0 if read_receipt(result)["metadata"]["status"] in ("numerically_verified", "unsupported") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="real TileOPs repository")
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ("prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--op", required=True)
        p.add_argument("--case-id", required=True)
        p.add_argument("--library", required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "run":
            p.add_argument("--device-launch-json", required=True)
            p.add_argument("--owner", required=True)
            p.add_argument("--guard-script", type=Path, help="existing host npu_guard.py; always launched -I -S -B")
            p.add_argument("--wait", type=int, default=600)
            p.add_argument("--timeout", type=int, default=120)
            p.add_argument("--native-evidence", type=Path, action="append", default=[])
    p = sub.add_parser("device")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--executed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = configure(args.root)
    if args.phase == "device":
        return device_capture(root, args.prepared, args.output)
    # CPU launcher sets this only after enforcing -I -S, disabling autoload,
    # installing an import blocker and selecting the isolated CPU torch package.
    if os.environ.get("ASCEND_PAIR_CPU_ISOLATED") != "1":
        raise RuntimeError("CPU phases must use ascend_pair_cpu.py under python -I -S -B")
    if args.phase == "prepare":
        print(prepare(root, args.op, args.case_id, args.library, args.output), flush=True)
        return 0
    if args.phase == "run":
        return guarded_run(root, args)
    receipt = compare(root, args.prepared, args.executed, args.output)
    print(receipt, flush=True)
    return 0 if read_receipt(receipt)["metadata"]["status"] in ("numerically_verified", "unsupported") else 1


if __name__ == "__main__":
    raise SystemExit(main())
