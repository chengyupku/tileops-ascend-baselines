"""Restore existing public Conv2DV2 + Swish composition permitted by user."""
import dataclasses
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]
from benchmarks.ascend_all_pairs.ops_nn.common import candidate, new_binding, op_name
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import source_hashes
def causal_convolution(case):
    import torch
    from dataclasses import replace
    binding=new_binding();p=case.params;name=op_name(case)
    def conv(xx,weight,bias):
        # One spatial dimension lifted to NCHW, C depthwise groups.
        x4=xx.float().contiguous().unsqueeze(-2);w4=weight.float().contiguous().reshape(weight.shape[0],1,1,weight.shape[-1])
        shape=(xx.shape[0],xx.shape[1],1,xx.shape[-1]-weight.shape[-1]+1)
        y=torch.empty(shape,device=xx.device,dtype=torch.float32)
        binding.call('Conv2DV2',[x4,w4,None if bias is None else bias.float().contiguous(),None],[y],
                     {'strides':[1,1,1,1],'pads':[0,0,0,0],'dilations':[1,1,1,1],
                      'groups':xx.shape[1],'data_format':'NCHW','pad_mode':'SPECIFIC','enable_hf32':False},
                     formats=[0,0,2,2,0])
        y=y.squeeze(-2)
        if p.get('activation')=='silu':
            z=torch.empty_like(y);binding.call('Swish',[y],[z],{'scale':1.0});y=z
        return y
    def run(x,weight,state,*args):
        if name=='CausalConv1dUpdateFwdOp':
            bias=args[0] if args else None
            shifted=torch.cat((state[...,1:],x.unsqueeze(-1)),dim=-1)
            y=conv(shifted,weight,bias).squeeze(-1)
            return y.to(x.dtype),shifted
        offsets=args[0].tolist();bias=args[1] if len(args)>1 else None
        flags=args[2].tolist() if len(args)>2 and args[2] is not None else None
        pieces=[];states=[];width=weight.shape[-1]
        for i,(begin,end) in enumerate(zip(offsets[:-1],offsets[1:])):
            history=state[i] if flags is not None and flags[i] else torch.zeros_like(state[i])
            tokens=torch.cat((history,x[:,begin:end]),dim=-1)
            pieces.append(conv(tokens.unsqueeze(0),weight,bias).squeeze(0))
            states.append(tokens[:,-(width-1):] if width>1 else tokens[:,:0])
        return torch.cat(pieces,dim=-1).to(x.dtype),torch.stack(states)
    result=candidate(binding,run,'Conv2DV2'+(' + Swish' if p.get('activation')=='silu' else ''),
                     'Read ragged offsets and state flags afresh; concatenate copied causal history, run original depthwise convolution and optional Swish, return full copied updated history. FP32 widening follows manifest reference; no torch convolution/math fallback.',
                     'conv/conv2d_v2','activation/swish')
    return replace(result,timing_mode='host_roundtrip' if name=='CausalConv1dFwdOp' else 'device_async')


def factory(case, library):
    if library != 'ops-nn' or case._call.signature.name not in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp'):
        raise ValueError('Only original ops-nn causal convolution composition')
    def build():
        c = causal_convolution(case)
        return dataclasses.replace(c, source_sha256={**c.source_sha256, **source_hashes(Path(__file__))})
    return build

if __name__ == '__main__':
    registry.candidate_factory = factory
    runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'), run_name='__main__')
