"""Diagnostic copies only; never change tensors passed to original callables."""
from pathlib import Path
import hashlib,json,os,time,sys

def capture(torch,label,value):
    root=Path.cwd()/'capture';root.mkdir(exist_ok=True)
    def host(x):
        if isinstance(x,torch.Tensor):return x.detach().to('cpu').clone()
        if isinstance(x,tuple):return tuple(map(host,x))
        if isinstance(x,list):return list(map(host,x))
        if isinstance(x,dict):return {k:host(v) for k,v in x.items()}
        return x
    def meta(x,original):
        if isinstance(x,torch.Tensor):
            raw=x.contiguous().view(torch.uint8).numpy().tobytes()
            return dict(shape=list(x.shape),dtype=str(x.dtype),source_device=str(original.device),stride=list(original.stride()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if isinstance(x,(tuple,list)):return [meta(a,b) for a,b in zip(x,original,strict=True)]
        if isinstance(x,dict):return {k:meta(v,original[k]) for k,v in x.items()}
        return x
    copied=host(value);p=root/(label+'.pt');assert not p.exists(),label
    torch.save(copied,p)
    entry=dict(label=label,time=time.time(),payload=str(p),payload_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),tensors=meta(copied,value))
    (root/(label+'.json')).write_text(json.dumps(entry,indent=2))
    with (root/'stages.jsonl').open('a') as f:f.write(json.dumps(dict(stage=label,time=time.time()))+'\n')
    print('CAPTURE',label,flush=True)

def install_native_trace(work):
    target='/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py'
    stream=(work/'native_calls.jsonl').open('x')
    def trace(frame,event,arg):
        if event in ('call','return') and frame.f_code.co_filename==target and frame.f_code.co_name=='chunk_gated_delta_rule_native':
            d=dict(event=event,time=time.time(),file=target,source_sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest(),function=frame.f_code.co_name)
            if event=='call':
                d['inputs']={k:dict(shape=list(v.shape),dtype=str(v.dtype),device=str(v.device)) for k,v in frame.f_locals.items() if hasattr(v,'shape') and hasattr(v,'device')}
            stream.write(json.dumps(d)+'\n');stream.flush()
    sys.setprofile(trace)
