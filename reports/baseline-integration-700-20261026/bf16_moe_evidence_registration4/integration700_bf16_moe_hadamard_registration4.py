"""CPU-lazy exact-key preparation for four audited BF16 MoE/Hadamard routes.

This module deliberately performs no device import, library load, compile, or numerical
replay.  It is an unregistered preparation artifact; ``run`` remains fail-closed on CPU.
"""
import builtins, hashlib, json, os, tempfile, types
from pathlib import Path
from types import MappingProxyType
ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "adapter.py"
ADAPTER_SHA256 = "4d843e71be9049844242420664c73df3907912edc4561ce6e42ff294bc4d0bb0"
ENV = MappingProxyType({"BF16_MOE_REG4_GUARD": "1"})
CASE_PARAMS = MappingProxyType({('ops-transformer', 'MoEGroupedGemmFwdOp', 'qwen3-30b-ep8-decode-bfloat16'): {'a': [[8, 2048], 'bfloat16'], 'b': [[16, 1536, 2048], 'bfloat16'], 'layout_metadata': [[16], 'int32'], 'output': [[8, 1536], 'bfloat16'], 'layout': {'contiguous': {'packing': 'tight', 'metadata_kind': 'physical_psum', 'alignment': 1}}, 'activation': None, 'out_dtype': None}, ('ops-transformer', 'MoEGroupedGemmFwdOp', 'qwen3-30b-ep8-routed256-bfloat16'): {'a': [[256, 2048], 'bfloat16'], 'b': [[16, 1536, 2048], 'bfloat16'], 'layout_metadata': [[16], 'int32'], 'output': [[256, 1536], 'bfloat16'], 'layout': {'contiguous': {'packing': 'tight', 'metadata_kind': 'physical_psum', 'alignment': 1}}, 'activation': None, 'out_dtype': None}, ('ops-transformer', 'HadamardTransformFwdOp', 'llama-8b-prefill-bfloat16'): {'x': [[4096, 4096], 'bfloat16'], 'output': [[4096, 4096], 'bfloat16'], 'base_order': 1}, ('ops-transformer', 'HadamardTransformFwdOp', 'llama3-head-dim-bfloat16'): {'x': [[262144, 128], 'bfloat16'], 'output': [[262144, 128], 'bfloat16'], 'base_order': 1}})
ENVELOPES = MappingProxyType({'qwen3-30b-ep8-decode-bfloat16': {'host_required': 5370544128, 'device_required': 3759013888, 'disk_required': 1476624384}, 'qwen3-30b-ep8-routed256-bfloat16': {'host_required': 5427429376, 'device_required': 3787456512, 'disk_required': 1483735040}, 'llama-8b-prefill-bfloat16': {'host_required': 3221225472, 'device_required': 2684354560, 'disk_required': 1207959552}, 'llama3-head-dim-bfloat16': {'host_required': 4294967296, 'device_required': 3221225472, 'disk_required': 1342177280}})
MANIFEST = MappingProxyType({str(ADAPTER): ADAPTER_SHA256})

def _checked(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError("Frozen dependency changed: " + str(path))
    return raw

def _snapshot():
    return {p: _checked(p, d) for p, d in MANIFEST.items()}

def _load_private(data=None):
    data = _snapshot() if data is None else data
    mod = types.ModuleType("_bf16_moe_reg4_private_adapter")
    mod.__file__ = str(ADAPTER)
    exec(compile(data[str(ADAPTER)], str(ADAPTER), "exec"), mod.__dict__)
    return mod

def _key(case, baseline="ops-transformer"):
    return (baseline, case._call.signature.name, case.id)

def matches(case, baseline="ops-transformer"):
    try:
        return _key(case, baseline) in CASE_PARAMS and json.dumps(case.params, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(CASE_PARAMS[_key(case, baseline)], sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (AttributeError, TypeError, ValueError):
        return False

def _guard_identity():
    return (os.environ.get("BF16_MOE_REG4_GUARD") == "1", os.getpid())

def _prepare_admission(key):
    if not _guard_identity()[0]:
        raise RuntimeError("foreground guard/resource admission required before input materialization")
    _snapshot()
    root = Path(os.environ.get("BF16_MOE_REG4_RUN_ROOT", str(ROOT / "runs"))).resolve()
    if root == ROOT or ROOT in root.parents:
        raise RuntimeError("fresh run root required")
    run = Path(tempfile.mkdtemp(prefix="bf16-moe-reg4-", dir=str(root)))
    (run / "RESOURCE_ADMISSION.json").write_text(json.dumps({"stage":"before_input_allocation","key":list(key),"envelope":ENVELOPES[key[2]],"cpu_only":True}, indent=2))
    return run

def make(case, baseline="ops-transformer"):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case, baseline):
        raise UnsupportedCase("Exact BF16 MoE/Hadamard key and canonical parameters required")
    _snapshot()
    key = _key(case, baseline)
    prepared = _prepare_admission(key) if _guard_identity()[0] else None
    def run(*args, **kwargs):
        nonlocal prepared
        if not matches(case, baseline):
            raise UnsupportedCase("Case changed after construction")
        if prepared is None:
            prepared = _prepare_admission(key)
        else:
            prepared, old = _prepare_admission(key), prepared
            del old
        # Deliberately no torch/torch_npu/ACL/CDLL import in this CPU artifact.
        raise RuntimeError("CPU preparation artifact: native device execution is not enabled")
    impl = types.SimpleNamespace(run=run, args=None, reset=None, noncomparable_reason="CPU-only registration preparation; no device replay")
    return Candidate(impl, "ops-transformer::aclnnGroupedMatmulV5", "Four exact BF16 MoE/Hadamard routes; original native attribution audited, execution deferred.", dict(MANIFEST), limitations=("No NPU execution in this artifact.", "Historical proc/maps and exact GetWorkspaceSize symbol resolution remain unproven."), timing_mode="host_roundtrip")

def wrap(previous):
    def dispatch(case):
        if _key(case) in CASE_PARAMS:
            return make(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase("No exact BF16 MoE/Hadamard route")
    dispatch.previous = previous
    return dispatch
