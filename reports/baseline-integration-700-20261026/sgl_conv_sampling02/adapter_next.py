"""Original SGL rejection/categorical kernels with explicit RNG uniforms."""
from pathlib import Path


def _build(case,categorical=False):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/sample/chain_speculative_sampling.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.sample.chain_speculative_sampling','chain_speculative_sampling_triton',host)
    def chain(draft,ids,target,seed,offset):
        batch,n,vocab=draft.shape
        # Original API explicitly accepts uniforms. This is RNG preparation,
        # not the candidate's acceptance, residual, or categorical computation.
        gen=torch.Generator(device=target.device)
        key=(int(seed.item()) ^ (int(offset.item()) * 0x9E3779B97F4A7C15)) & ((1<<63)-1)
        gen.manual_seed(key)
        uniforms=torch.rand((batch,n+1),device=target.device,dtype=torch.float32,generator=gen)
        candidates=torch.cat((torch.zeros((batch,1),device=ids.device,dtype=torch.int64),ids.to(torch.int64)),dim=1).contiguous()
        retrieve=torch.arange(batch*(n+1),device=ids.device,dtype=torch.int64).reshape(batch,n+1)
        predicts=torch.full((batch*(n+1),),-1,device=ids.device,dtype=torch.int32)
        accepted_indices=torch.full((batch,n+1),-1,device=ids.device,dtype=torch.int32)
        accepted=torch.empty(batch,device=ids.device,dtype=torch.int32)
        dummy=torch.empty(0,device=ids.device,dtype=torch.int64)
        dd=draft.contiguous().float() if n else torch.zeros((batch,1,vocab),device=target.device,dtype=torch.float32)
        fn(predicts,accepted_indices,accepted,candidates,retrieve,dummy,dummy,uniforms.contiguous(),uniforms[:,-1].contiguous(),target.contiguous().float(),dd)
        return predicts.reshape(batch,n+1),accepted
    def draw(probs,seed,offset):
        b,v=probs.shape
        tokens,_=chain(torch.empty((b,0,v),device=probs.device,dtype=probs.dtype),torch.empty((b,0),device=probs.device,dtype=torch.int32),probs[:,None],seed,offset)
        return tokens[:,0]
    return Candidate(Implementation(draw if categorical else chain),'sgl_kernel_npu.sample.chain_speculative_sampling.chain_speculative_sampling_triton',
                     'Map draft ids to root-prefixed linear-chain metadata and native output slots; no-draft specialization samples categorical probabilities. Generate deterministic seed/offset-keyed uniforms per call; SGL performs every rejection, residual reduction and inverse-CDF sample.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),
                     limitations=('Uses its own RNG stream as permitted by unchanged statistical workload verifier; RNG preparation and host scalar transfers count.',),timing_mode='host_roundtrip')


def chain(case):return _build(case,False)
def categorical(case):return _build(case,True)


def conv(case):
    import importlib,sys,torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,source_hashes
    root=LIBRARIES/'sgl-kernel-npu'
    package=root/'python/sgl_kernel_npu'
    sys.path.insert(0,str(package))
    module=importlib.import_module('sgl_kernel_npu')
    assert Path(module.__file__).resolve()==package/'sgl_kernel_npu/__init__.py'
    def run(x,weight,bias):
        n,c,h,w=x.shape
        outc=weight.shape[0]
        assert tuple(weight.shape[2:])==(1,1)
        matrix=x.permute(0,2,3,1).contiguous().view(-1,c)
        wt=weight.reshape(1,outc,c).contiguous()
        y=torch.zeros((matrix.shape[0],outc),dtype=x.dtype,device=x.device) if bias is None else bias.expand(matrix.shape[0],outc).contiguous().clone()
        indices=torch.zeros(1,dtype=torch.int32,device=x.device)
        ranks=torch.tensor([c],dtype=torch.int32,device=x.device)
        offsets=torch.tensor([0,outc],dtype=torch.int32,device=x.device)
        lengths=torch.tensor([1],dtype=torch.int32,device=x.device)
        # Independent spatial vectors; each invocation retains full K and Cout.
        for i in range(matrix.shape[0]):
            torch.ops.npu.sgemmc_expand(matrix[i:i+1],wt,indices,lengths,ranks,offsets,y[i:i+1])
            torch.npu.synchronize()
        return y.reshape(n,h,w,outc).permute(0,3,1,2).contiguous()
    sources=[p for p in (root/'csrc/lora').rglob('*') if p.suffix in ('.cpp','.h','.hpp')]
    return Candidate(Implementation(run),'torch.ops.npu.sgemmc_expand',
        'Bijective NCHW spatial-row layout, full original channel product per spatial position; original residual input supplies bias; no kernel changes.',
        source_hashes(Path(__file__),package/'sgl_kernel_npu/__init__.py',package/'sgl_kernel_npu/lib/libsgl_kernel_npu.so',root/'csrc/pytorch_extensions.cpp',*sources))
PROVIDERS={'Conv2dFwdOp':conv,'ChainSpeculativeSamplingFwdOp':chain}
