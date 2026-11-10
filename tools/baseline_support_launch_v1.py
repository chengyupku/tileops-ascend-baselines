"""Versioned, top-level-only candidates for the 2026-10-29 support sweep.

This file is invoked inside the existing guard and original cache measurement.
It neither edits installed libraries nor changes the checker or timing core.
Old providers and their successful cache fingerprints remain unchanged.
"""
import dataclasses
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]


def install():
    from benchmarks.ascend_all_pairs import registry
    from benchmarks.ascend_baselines import source_hashes, UnsupportedCase
    from benchmarks.api import Implementation
    original = registry.candidate_factory

    def factory(case, library):
        op = case._call.signature.name

        def build():
            if library == 'sgl-kernel-npu' and op == 'GDNFwdOp' and str(case.inputs[0].dtype) == 'torch.float16':
                from benchmarks.ascend_all_pairs.sgl.linear import _triton_gdn
                candidate = _triton_gdn(case)
                native = candidate.implementation

                def run(q, k, v, g, *rest):
                    return native.run(q, k, v, g.float(), *rest)

                candidate = dataclasses.replace(candidate,
                    implementation=dataclasses.replace(native, run=run),
                    adaptation=candidate.adaptation + ' Gate widens losslessly to FP32 at the original public API; Q/K/V unchanged, cast included in timing.')
            elif library == 'ops-nn':
                from benchmarks.ascend_all_pairs.ops_nn import staging
                providers = dict(staging.CANDIDATES)
                providers.update({name: staging.variance for name in ('VarFwdOp', 'VarMeanFwdOp', 'StdFwdOp')})
                providers.update({name: staging.clamp for name in ('ClampScalarFwdOp', 'ClampTensorFwdOp', 'HardtanhFwdOp')})
                providers.update({name: staging.boolean_logic for name in ('LogicalAndFwdOp', 'LogicalOrFwdOp', 'LogicalNotFwdOp')})
                providers['BitwiseNotFwdOp'] = staging.bitwise_not
                providers.update({name: staging.ada_layer_norm for name in ('AdaLayerNormFwdOp', 'AdaLayerNormZeroFwdOp')})
                if op not in providers:
                    raise UnsupportedCase('No existing ops-nn top-level candidate for ' + op)
                candidate = providers[op](case)
            else:
                candidate = original(case, library)()
            return dataclasses.replace(candidate, source_sha256={
                **candidate.source_sha256, **source_hashes(Path(__file__))})

        return build

    registry.candidate_factory = factory


if __name__ == '__main__':
    install()
    runpy.run_path(str(ROOT / 'tools/measure_baseline_cache_pair.py'), run_name='__main__')
