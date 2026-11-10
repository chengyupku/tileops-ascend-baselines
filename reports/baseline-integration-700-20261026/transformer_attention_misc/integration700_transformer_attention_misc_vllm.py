from pathlib import Path

def cumsum(case):
    import importlib
    import torch
    from vllm_private.runtime import kda_module,candidate
    package,sources=kda_module();native=importlib.import_module(package.__package__+'.cumsum')
    dim=case.params['dim']%case.inputs[0].ndim
    def run(x):
        data=x.movedim(dim,-1).contiguous();shape=data.shape;width=shape[-1]
        y=native.chunk_local_cumsum_vector(data.reshape(-1,width,1,1),chunk_size=1<<(width-1).bit_length(),reverse=False,output_dtype=x.dtype)
        return y.reshape(shape).movedim(-1,dim).contiguous()
    return candidate(run,'kda.cumsum.chunk_local_cumsum_vector','Complete scan axis as one native chunk; unchanged original input and output dtypes; restore layout.',[Path(__file__),*sources])

def linear(case):
    from vllm_private.linear_attention import linear_attention
    return linear_attention(case)
