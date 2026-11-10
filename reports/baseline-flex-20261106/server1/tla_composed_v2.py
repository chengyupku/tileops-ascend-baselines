"""Public original TLA operator composition; no kernel/compiler edits."""
import dataclasses
from math import prod
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
ROOT=Path('/opt/tileops-baseline-cache/workspace')
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.tla import adapters
from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
from benchmarks.api import Implementation
OPS={'SoftplusFwdOp','HardswishFwdOp','ClampTensorFwdOp','LogicalAndFwdOp','LogicalOrFwdOp','MeanFwdOp','VarFwdOp','StdFwdOp','VarMeanFwdOp','LogSumExpFwdOp','LogSoftmaxFwdOp','ArgmaxFwdOp','LerpScalarFwdOp','HardsigmoidFwdOp'}

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
        if op == 'SoftplusFwdOp':
            if x.dtype != torch.bfloat16:
                raise UnsupportedCase('Original GDN gating input is fixed BF16; no lossy FP16 conversion')
            from benchmarks.ascend_all_pairs.tla import native
            import importlib.util
            native.runtime()
            source='examples/xllm_kernels/fused_gdn_gating.py'
            directory=native.VENDOR/'examples/xllm_kernels'
            saved=sys.modules.get('utils');sys.path.insert(0,str(directory))
            try:
                spec_u=importlib.util.spec_from_file_location('utils',directory/'utils.py')
                utils=importlib.util.module_from_spec(spec_u);sys.modules['utils']=utils;spec_u.loader.exec_module(utils)
                spec=importlib.util.spec_from_file_location('flex_original_gdn_gating',directory/'fused_gdn_gating.py')
                module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            finally:
                sys.path.remove(str(directory))
                if saved is None:sys.modules.pop('utils',None)
                else:sys.modules['utils']=saved
            shape=tuple(p['output'][0]);size=prod(shape);heads=128
            if size%heads:raise UnsupportedCase('Exact reshape requires multiple128 elements')
            rows=size//heads
            module.pass_configs=module.DEFAULT_ASCEND_PASS_CONFIGS
            kernel=module.fused_gdn_gating_kernel_jit(num_batches=rows,compile_max_batch=rows,num_heads=heads)
            def gating(x):
                a=x.contiguous().reshape(rows,heads)
                zero=torch.zeros((heads,),dtype=torch.float32,device=x.device)
                b=torch.zeros_like(a);g=torch.empty((rows,heads),dtype=torch.float32,device=x.device);beta_out=torch.empty_like(a)
                kernel(zero,a,b,zero,g,beta_out,rows,float(p.get('beta',1.)),float(p.get('threshold',20.)))
                return g.reshape(shape)
            gate=native.candidate(gating,source+':build_fused_gdn_gating_kernel','BF16 exact reshape; neutral A_log/dt_bias; original negative softplus output; all constants and allocations timed.',source,'examples/xllm_kernels/utils.py')
            parts.append(gate);neg=ew('NegFwdOp',shape)
            def run(x):return neg(gate.implementation.run(x)).to(x.dtype)
        elif op == 'HardswishFwdOp':
            from benchmarks.ascend_all_pairs.tla import native
            shape=tuple(p['output'][0]);size=prod(shape);rows=((size+63)//64+63)//64*64;cols=64
            source='testing/python/language/test_tilelang_ascend_language_elementwise.py'
            module=native.load(source,'vec_mul_add_dst','pass_configs')
            kernel=native.compile_builder(module,'vec_mul_add_dst',rows,cols,64,64,dtype='float32',out_idx=[-1])
            def affine(x):
                v=adapters._flat_pack(x,shape,rows,cols,torch.float32)
                scale=torch.full((rows,cols),1./6.,dtype=torch.float32,device=x.device)
                bias=torch.full((rows,cols),0.5,dtype=torch.float32,device=x.device)
                return kernel(v,scale,bias).reshape(-1)[:size].reshape(shape)
            aff=native.candidate(affine,source+':vec_mul_add_dst','Original multiply-add with broadcast constants; all packing and constants inside run.',source)
            parts.append(aff);clamp=ew('ClampScalarFwdOp',shape,min=0.,max=1.);mul=ew('MulFwdOp',shape)
            def run(x):return mul(x.float(),clamp(aff.implementation.run(x))).to(x.dtype)
        elif op == 'ClampTensorFwdOp':
            shape=tuple(p['output'][0]);lo=ew('MaximumFwdOp',shape) if 'min' in p else None;hi=ew('MinimumFwdOp',shape) if 'max' in p else None
            def run(x,lower=None,upper=None):
                value=x.float()
                if lo is not None:value=lo(value,lower.float())
                if hi is not None:value=hi(value,upper.float())
                return value.to(x.dtype)
        elif op in {'LogicalAndFwdOp','LogicalOrFwdOp'}:
            shape=tuple(p['output'][0])
            cp={'input':[list(shape),'float32'],'other':[list(shape),'float32'],'output':[list(shape),'bool']}
            comparison=adapters.elementwise('NeFwdOp',SimpleNamespace(params=cp));parts.append(comparison)
            bp={'input':[list(shape),'bool'],'other':[list(shape),'bool'],'output':[list(shape),'bool']}
            logical=adapters.elementwise(op,SimpleNamespace(params=bp));parts.append(logical)
            def run(x,y):
                zero=torch.zeros(shape,dtype=torch.float32,device=x.device)
                a=comparison.implementation.run(x.float(),zero)
                b=comparison.implementation.run(y.float(),zero)
                return logical.implementation.run(a,b)
        elif op in {'LerpScalarFwdOp','HardsigmoidFwdOp'}:
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
        sources[str(Path(__file__).resolve())] = __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
        return Candidate(Implementation(run),'original tilelang-ascend public operator composition for '+op,
                         'Existing whole native operators; dtype/layout/constant broadcasting and native composition only. All adaptation included in timing. No kernel or compiler changes.',sources)
    return build
