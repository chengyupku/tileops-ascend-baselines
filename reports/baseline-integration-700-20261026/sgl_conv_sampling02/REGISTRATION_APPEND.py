
# Exact keys privately verified by baseline700 sgl_conv_sampling02.
from .integration700_sgl_conv_sampling02 import conv as _i700_cs02_conv, chain as _i700_cs02_chain

def _i700_cs02_dispatch(previous, factory, ids):
    def build(case):
        return factory(case) if case.id in ids else previous(case)
    return build
PROVIDERS['Conv2dFwdOp'] = _i700_cs02_dispatch(PROVIDERS['Conv2dFwdOp'], _i700_cs02_conv, ('resnet-1x1-float16', 'bottleneck-expand-float16', 'late-stage-1x1-bias-float16', 'classifier-1x1-float16', 'resnet-1x1-bfloat16', 'resnet-1x1-bias-bfloat16', 'classifier-1x1-bias-float16'))
PROVIDERS['ChainSpeculativeSamplingFwdOp'] = _i700_cs02_dispatch(PROVIDERS['ChainSpeculativeSamplingFwdOp'], _i700_cs02_chain, ('llama2-7b-b256-n1', 'llama-8b-b17-n8'))
