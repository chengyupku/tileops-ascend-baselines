"""Private registration candidate: one original mode6 full attention per run."""
from pathlib import Path
import hashlib,json,os,time
import adapter
P=Path(__file__).resolve().parent

def event(stage,**fields):
    with (P/(os.environ['CASE_ID']+'.native.jsonl')).open('a') as f:
        f.write(json.dumps(dict(stage=stage,time=time.time(),**fields))+'\n')

def provider(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    import importlib
    import cann_ops_transformer as cot
    assert case.id in adapter.CASES
    api=importlib.import_module('cann_ops_transformer.ops.quant_flash_attn')
    event('original_extension_build_start')
    mod=api.quant_flash_attn_op_builder.load()
    artifact=Path(mod.__file__)
    event('original_extension_before_first_call',path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    for name in ('quant_flash_attn_metadata','quant_flash_attn'):
        original=getattr(cot,name)
        def observed(*args,_name=name,_fn=original,**kwargs):
            event('native_api_enter',api=_name)
            result=_fn(*args,**kwargs)
            event('native_api_host_return',api=_name)
            return result
        setattr(cot,name,observed)
    def run(q,k,v,cuq,cuk,qs,ks,vs,rope_cos=None,rope_sin=None,sinks=None):
        assert rope_cos is None and rope_sin is None and sinks is None
        return adapter.run(case.id,q,k,v,cuq,cuk,qs,ks,vs,sm_scale=case.params.get('sm_scale'))
    identity=json.loads((P/'PRE_USE_IDENTITY.json').read_text())
    return Candidate(Implementation(run), 'original cann_ops_transformer.quant_flash_attn mode6 + original metadata',
                     'Batch lifted to independent head ranges; exact FP32 scale copies; paging/zero padding/output selection all in run.',
                     {k:v['sha256'] for k,v in identity.items()},limitations=('B-fold head expansion; unvalidated until original checker passes',),timing_mode='host_sync')
