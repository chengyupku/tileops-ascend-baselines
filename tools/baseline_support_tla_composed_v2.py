"""Public original TLA operator composition; no kernel/compiler edits."""
import dataclasses
from math import prod
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.tla import adapters
from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
from benchmarks.api import Implementation
OPS={'MeanFwdOp','VarFwdOp','StdFwdOp','VarMeanFwdOp','LogSumExpFwdOp','LogSoftmaxFwdOp','ArgmaxFwdOp','LerpScalarFwdOp','HardsigmoidFwdOp','HardswishFwdOp','PreluFwdOp','EluFwdOp','SeluFwdOp'}

def factory(case,library):
    op=case._call.signature.name
    if library!='tilelang-ascend' or op not in OPS:raise UnsupportedCase('No bounded public composition for '+op)
    def build():
        import torch
        p=case.params;x=case.inputs[0];parts=[]
        if x.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise UnsupportedCase('Original float32 public composition has no exact integer/FP64 input route')
        def ew(name,shape,**extra):
            params={'input':[list(shape),'float32'],'other':[list(shape),'float32'],'output':[list(shape),'float32'],**extra}
            c=adapters.elementwise(name,SimpleNamespace(params=params));parts.append(c);return c.implementation.run
        if op in {'LerpScalarFwdOp','HardsigmoidFwdOp'}:
            shape=tuple(p['output'][0]);add=ew('AddFwdOp',shape)
            if op=='LerpScalarFwdOp':
                sub=ew('SubFwdOp',shape);mul=ew('MulFwdOp',shape)
                def run(x,end):
                    a=x.float().expand(shape);b=end.float().expand(shape)
                    return add(a,mul(sub(b,a),torch.full(shape,p['weight'],device=x.device))).to(x.dtype)
            else:
                clamp=ew('ClampScalarFwdOp',shape,min=0.,max=6.);div=ew('DivFwdOp',shape)
                def run(x):
                    return div(clamp(add(x.float(),torch.full(shape,3.,device=x.device))),torch.full(shape,6.,device=x.device)).to(x.dtype)
        elif op in {'HardswishFwdOp','PreluFwdOp','EluFwdOp','SeluFwdOp'}:
            shape=tuple(p['output'][0]);add=ew('AddFwdOp',shape);mul=ew('MulFwdOp',shape)
            def const(x,v):return torch.full(shape,v,dtype=torch.float32,device=x.device)
            if op=='HardswishFwdOp':
                clamp=ew('ClampScalarFwdOp',shape,min=0.,max=6.);div=ew('DivFwdOp',shape)
                def run(x):
                    v=x.float();return div(mul(v,clamp(add(v,const(x,3.)))),const(x,6.)).to(x.dtype)
            else:
                maximum=ew('MaximumFwdOp',shape);minimum=ew('MinimumFwdOp',shape)
                if op=='PreluFwdOp':
                    def run(x,weight):
                        v=x.float();w=weight.float()
                        if w.numel()!=1:w=w.reshape((1,w.numel())+(1,)*(x.ndim-2))
                        return add(maximum(v,const(x,0.)),mul(minimum(v,const(x,0.)),w.expand(shape))).to(x.dtype)
                else:
                    exp=ew('ExpFwdOp',shape);sub=ew('SubFwdOp',shape)
                    alpha=p.get('alpha',1.) if op=='EluFwdOp' else 1.6732632423543772
                    scale=1. if op=='EluFwdOp' else 1.0507009873554805
                    def run(x):
                        v=x.float();negative=mul(sub(exp(minimum(v,const(x,0.))),const(x,1.)),const(x,alpha))
                        return mul(add(maximum(v,const(x,0.)),negative),const(x,scale)).to(x.dtype)
        elif op=='ArgmaxFwdOp':
            shape=tuple(x.shape);neg=ew('NegFwdOp',shape)
            params={**p,'x':[list(shape),'float32']}
            c=adapters.argmin(SimpleNamespace(params=params));parts.append(c)
            def run(x):return c.implementation.run(neg(x.float()))
        else:
            shape=tuple(x.shape);perm,rows,width=adapters._axis_layout(shape,p.get('dim',-1))
            if width>32768:raise UnsupportedCase('Original complete-row reduction exceeds32768; stop complex multi-stage reduction under bounded-effort policy')
            matrix=(rows,width);column=(rows,1)
            def reduce(name):
                params={'x':[list(matrix),'float32'],'output':[list(column),'float32'],'dim':-1,'keepdim':True}
                c=adapters.reduction(name,SimpleNamespace(params=params));parts.append(c);return c.implementation.run
            summation=reduce('SumFwdOp');divide=ew('DivFwdOp',column)
            output_spec=p.get('output',p.get('var'));out_shape=tuple(output_spec[0]);out_dtype=getattr(torch,output_spec[1])
            if op in {'MeanFwdOp','VarFwdOp','VarMeanFwdOp','StdFwdOp'}:
                if op!='MeanFwdOp':sub=ew('SubFwdOp',matrix);mul=ew('MulFwdOp',matrix)
                if op=='StdFwdOp':sqrt=ew('SqrtFwdOp',column)
                correction=p.get('correction',1)
                if width<=correction:raise UnsupportedCase('Nonpositive variance denominator outside this simple composition')
                def run(x):
                    v=x.permute(perm).contiguous().reshape(matrix).float()
                    mean=divide(summation(v),torch.full(column,float(width),device=x.device))
                    if op=='MeanFwdOp':return mean.reshape(out_shape).to(out_dtype)
                    centered=sub(v,mean.expand(matrix));squared=mul(centered,centered)
                    var=divide(summation(squared),torch.full(column,float(width-correction),device=x.device))
                    if op=='VarMeanFwdOp':return var.reshape(out_shape).to(out_dtype),mean.reshape(tuple(p['mean'][0])).to(getattr(torch,p['mean'][1]))
                    if op=='StdFwdOp':var=sqrt(var)
                    return var.reshape(out_shape).to(out_dtype)
            else:
                maximum=reduce('AmaxFwdOp');sub=ew('SubFwdOp',matrix);exp=ew('ExpFwdOp',matrix);log=ew('LogFwdOp',column);add=ew('AddFwdOp',column)
                inverse=[perm.index(i) for i in range(len(shape))]
                def run(x):
                    v=x.permute(perm).contiguous().reshape(matrix).float();m=maximum(v)
                    lse=add(log(summation(exp(sub(v,m.expand(matrix))))),m)
                    if op=='LogSumExpFwdOp':return lse.reshape(out_shape).to(out_dtype)
                    return sub(v,lse.expand(matrix)).reshape(tuple(shape[i] for i in perm)).permute(inverse).contiguous().to(out_dtype)
        sources={}
        for part in parts:sources.update(part.source_sha256)
        sources.update(source_hashes(Path(__file__)))
        return Candidate(Implementation(run),'original tilelang-ascend public operator composition for '+op,
                         'Existing whole native operators; dtype/layout/constant broadcasting and native composition only. All adaptation included in timing. No kernel or compiler changes.',sources)
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
