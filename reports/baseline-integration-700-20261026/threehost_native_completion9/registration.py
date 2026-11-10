"""Pending parent publication. CPU-lazy exact contracts for the tested ARM deployment."""
from pathlib import Path
import json
import importlib.util
import hashlib
P=Path(__file__).resolve().parent

def _normalize(x):
    return json.loads(json.dumps(x,default=str))

def matches(case):
    contracts=json.loads((P/'CONTRACTS.json').read_text())
    key=(case._call.signature.name,case.id)
    for c in contracts:
        if key==(c['op'],c['case_id']):
            return (_normalize(case.params)==c['parameters'] and
                    _normalize(case._call.tensors)==c['declared_tensors'] and
                    list(case._call.signature.inputs)==c['declared_input_names'] and
                    list(case._call.signature.outputs)==c['declared_output_names'])
    return False

def make_candidate(case):
    if not matches(case):
        raise ValueError('Not one of the exact reviewed native-completion9 contracts')
    mode='fp8' if case._call.signature.name=='BmmFP8FwdOp' else 'bf16arm2'
    frozen=json.loads((P/('FROZEN_'+mode+'.json')).read_text())
    # Require the actual tested host, bridge and deployment identity before loading native code.
    for name,expected in frozen['sha256'].items():
        if '/snapshots/' in name:continue
        path=Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Tested closure unavailable or changed: '+name)
    path=(P.parent/'transformer_dense/integration700_transformer_dense_fp8_bmm.py' if mode=='fp8'
          else P/'arm_closure/bf16_moe_hadamard_native10/adapter.py')
    spec=importlib.util.spec_from_file_location('completion9_original_'+mode,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return (module.fp8_bmm if mode=='fp8' else module.make_candidate)(case)

def wrap(previous):
    def factory(case):
        if matches(case):return make_candidate(case)
        if previous is None:raise ValueError('No previous provider for this case')
        return previous(case)
    return factory
