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
import importlib.util
_spec=importlib.util.spec_from_file_location('benchmarks.ascend_all_pairs.tla.old1076_adapters_v3',ROOT/'tools/baseline_old1076_tla_adapters_v3.py')
adapters=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapters)
from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
from benchmarks.api import Implementation
OPS={'MeanFwdOp','VarFwdOp','StdFwdOp','VarMeanFwdOp','LogSumExpFwdOp','LogSoftmaxFwdOp','ArgmaxFwdOp','ArgminFwdOp','LerpScalarFwdOp','HardsigmoidFwdOp','HardswishFwdOp','PreluFwdOp','EluFwdOp','SeluFwdOp','SoftplusFwdOp','MishFwdOp'}

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
        elif op in {'SoftplusFwdOp','MishFwdOp'}:
            shape=tuple(p['output'][0]);add=ew('AddFwdOp',shape);mul=ew('MulFwdOp',shape);div=ew('DivFwdOp',shape);maximum=ew('MaximumFwdOp',shape);absolute=ew('AbsFwdOp',shape);neg=ew('NegFwdOp',shape);exp=ew('ExpFwdOp',shape);log=ew('LogFwdOp',shape)
            beta=p.get('beta',1.) if op=='SoftplusFwdOp' else 1.
            if op=='MishFwdOp':tanh=ew('TanhFwdOp',shape)
            else:
                c=adapters.elementwise('GtFwdOp',SimpleNamespace(params={'input':[list(shape),'float32'],'other':[list(shape),'float32'],'output':[list(shape),'bool']}));parts.append(c);gt=c.implementation.run
                c=adapters.select('WhereFwdOp',SimpleNamespace(params={'input':[list(shape),'float32'],'output':[list(shape),'float32']}));parts.append(c);select=c.implementation.run
            def constant(x,v):return torch.full(shape,v,device=x.device,dtype=torch.float32)
            def run(x):
                v=x.float();z=mul(v,constant(x,beta));sp=div(add(maximum(z,constant(x,0.)),log(add(exp(neg(absolute(z))),constant(x,1.)))),constant(x,beta))
                if op=='MishFwdOp':return mul(v,tanh(sp)).to(x.dtype)
                return select(gt(z,constant(x,p.get('threshold',20.))),v,sp).to(x.dtype)
        elif op in ('ArgmaxFwdOp','ArgminFwdOp'):
            shape=tuple(x.shape);perm,rows,width=adapters._axis_layout(shape,p.get('dim'));matrix=(rows,width);column=(rows,1)
            if width>262144:raise UnsupportedCase('Bounded public block argmax exceeds262144')
            def reduction(name):
                if width<=4096:
                    c=adapters.reduction(name,SimpleNamespace(params={'x':[list(matrix),'float32'],'output':[list(column),'float32'],'dim':-1,'keepdim':True}));parts.append(c);return c.implementation.run
                blocks=(width+4095)//4096
                a=adapters.reduction(name,SimpleNamespace(params={'x':[[rows*blocks,4096],'float32'],'output':[[rows*blocks,1],'float32'],'dim':-1,'keepdim':True}))
                b=adapters.reduction(name,SimpleNamespace(params={'x':[[rows,blocks],'float32'],'output':[[rows,1],'float32'],'dim':-1,'keepdim':True}))
                parts.extend([a,b])
                def block_reduce(value):
                    pad=torch.nn.functional.pad(value,(0,blocks*4096-width),value=float('-inf') if name=='AmaxFwdOp' else float('inf'))
                    return b.implementation.run(a.implementation.run(pad.reshape(rows*blocks,4096)).reshape(rows,blocks))
                return block_reduce
            maximum=reduction('AmaxFwdOp' if op=='ArgmaxFwdOp' else 'AminFwdOp');minimum=reduction('AminFwdOp')
            c=adapters.elementwise('EqFwdOp',SimpleNamespace(params={'input':[list(matrix),'float32'],'other':[list(matrix),'float32'],'output':[list(matrix),'bool']}));parts.append(c);equal=c.implementation.run
            c=adapters.select('WhereFwdOp',SimpleNamespace(params={'input':[list(matrix),'float32'],'output':[list(matrix),'float32']}));parts.append(c);select=c.implementation.run
            def run(x):
                v=x.permute(perm).contiguous().reshape(matrix).float();indices=torch.arange(width,device=x.device,dtype=torch.float32).reshape(1,width).expand(matrix)
                masked=select(equal(v,maximum(v).expand(matrix)),indices,torch.full(matrix,float('inf'),device=x.device))
                return minimum(masked).long().reshape(tuple(p['output'][0]))
        else:
            shape=tuple(x.shape);perm,rows,width=adapters._axis_layout(shape,p.get('dim',-1))
            if width>262144:raise UnsupportedCase('Bounded public block reduction up to262144')
            matrix=(rows,width);column=(rows,1)
            def reduce(name):
                if width <= 4096:
                    params={'x':[list(matrix),'float32'],'output':[list(column),'float32'],'dim':-1,'keepdim':True}
                    c=adapters.reduction(name,SimpleNamespace(params=params));parts.append(c);return c.implementation.run
                blocks=(width+4095)//4096
                a=adapters.reduction(name,SimpleNamespace(params={'x':[[rows*blocks,4096],'float32'],'output':[[rows*blocks,1],'float32'],'dim':-1,'keepdim':True}))
                b=adapters.reduction(name,SimpleNamespace(params={'x':[[rows,blocks],'float32'],'output':[[rows,1],'float32'],'dim':-1,'keepdim':True}))
                parts.extend([a,b])
                def run_blocks(value):
                    padded=torch.nn.functional.pad(value,(0,blocks*4096-width),value=float('-inf') if name=='AmaxFwdOp' else float('inf') if name=='AminFwdOp' else 0.)
                    partial=a.implementation.run(padded.contiguous().reshape(rows*blocks,4096))
                    return b.implementation.run(partial.reshape(rows,blocks))
                return run_blocks
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
        sources.update(source_hashes(Path(__file__),ROOT/'tools/baseline_old1076_tla_adapters_v3.py'))
        return Candidate(Implementation(run),'original tilelang-ascend public operator composition for '+op,
                         'Existing whole native operators; dtype/layout/constant broadcasting and native composition only. All adaptation included in timing. No kernel or compiler changes.',sources)
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
