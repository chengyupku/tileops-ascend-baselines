"""Additional original-contract checks for layout changes, outside timing."""
import torch
from workloads.quantization import quantize as ref
from workloads.quantization import int8_dequant as dq
from workloads.numerics import compare_outputs
import adapters

def check(op,params,evidence,variant):
    if variant=='native':
        import native_adapters
        from workloads.quantization import int8_dequant as original
        dt=getattr(torch,params['out_dtype']); shape=(3,129); records=[]
        fn=native_adapters.factory(op,params)
        oracle={'INT8DequantPerBlockFwdOp':original.int8_dequant_per_block,'INT8DequantPerChannelFwdOp':original.int8_dequant_per_channel,'INT8DequantPerTensorFwdOp':original.int8_dequant_per_tensor}[op]
        for kind in ('random','zero','extremes','negative_scale'):
            q=torch.arange(387).remainder(256).sub(128).to(torch.int8).reshape(shape)
            ss=(3,2) if 'Block' in op else ((3,) if 'Channel' in op else (1,))
            scale=torch.rand(ss,dtype=torch.float32)*0.03
            if kind=='zero':scale.zero_()
            elif kind=='extremes':scale.fill_(0.5)
            elif kind=='negative_scale':scale.fill_(-0.25)
            expected=oracle(q,scale,dt);got=fn(q.npu(),scale.npu());torch.npu.synchronize()
            compare_outputs(got.cpu(),expected,evidence)
            records.append({'shape':shape,'kind':kind,'dtype':str(dt),'passed':True})
        return records
    if op not in ('INT8QuantPerBlockFwdOp','FP8QuantPerBlockFwdOp','INT8DequantPerBlockFwdOp','INT4QuantPerGroupFwdOp'):return []
    records=[];torch.manual_seed(1234)
    for dtype in (torch.float16,torch.bfloat16,torch.float32):
        if op=='INT4QuantPerGroupFwdOp':shape=(2,int(params['group_size'])*2)
        elif op=='FP8QuantPerBlockFwdOp':shape=(129,129)
        else:shape=(3,129)
        for kind in ('random','zero','constant','ties'):
            if op=='INT8DequantPerBlockFwdOp':
                q=torch.randint(-127,128,shape,dtype=torch.int8)
                scale=torch.rand(shape[0],(shape[1]+127)//128,dtype=torch.float32)*0.01
                if kind=='zero':q.zero_();scale.zero_()
                elif kind=='constant':q.fill_(-127);scale.fill_(1)
                elif kind=='ties':q.fill_(1);scale.fill_(0.5)
                xs=(q,scale);oracle=lambda q,s:dq.int8_dequant_per_block(q,s,dtype)
                fn=lambda q,s:adapters.dequant_block(q,s,dtype)
            else:
                x=torch.randn(shape,dtype=dtype)
                if kind=='zero':x.zero_();x.reshape(-1)[0]=-0.0
                elif kind=='constant':x.fill_(3)
                elif kind=='ties':x=torch.arange(x.numel()).remainder(17).reshape(shape).to(dtype)-8
                xs=(x,)
                if op=='INT8QuantPerBlockFwdOp':oracle=ref.int8_quant_per_block;fn=adapters.int8_block
                elif op=='FP8QuantPerBlockFwdOp':oracle=ref.fp8_quant_per_block;fn=adapters.fp8_block
                else:
                    oracle=lambda x:ref.int4_quant_per_group(x,params['group_size'])
                    fn=lambda x:adapters.int4_group(x,params['group_size'])
            expected=oracle(*xs)
            device_args=tuple(x.npu() for x in xs)
            if variant=='compile':fn=torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False)
            got=fn(*device_args);torch.npu.synchronize()
            if isinstance(got,tuple):got=tuple(x.cpu() for x in got)
            else:got=got.cpu()
            compare_outputs(got,expected,evidence)
            records.append({'dtype':str(dtype),'shape':shape,'kind':kind,'passed':True})
    return records
