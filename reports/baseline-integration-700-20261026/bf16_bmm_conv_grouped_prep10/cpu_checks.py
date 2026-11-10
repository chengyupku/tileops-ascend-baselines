"""Pure Python metadata mock tests. No torch/native/reference import or computation."""
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import yaml
import adapter

D = Path(__file__).resolve().parent
W = D.parents[2]
R = D.parent
checks = []
def record(name, fn):
    fn()
    checks.append({'name': name, 'status': 'pass'})
def require(v):
    if not v: raise AssertionError(v)
def rejects(fn, error=ValueError):
    try: fn()
    except error: return
    raise AssertionError('expected rejection')

def strides(shape):
    out = []; stride = 1
    for n in reversed(shape): out.append(stride); stride *= n
    return tuple(reversed(out))
class Tensor:
    def __init__(self, shape, dtype='bfloat16', name='tensor', stride=None, values=None, trace=()):
        self.shape = tuple(shape); self.dtype=dtype; self.device='mock'
        self.name=name; self._stride=stride or strides(shape); self.values=values
        self.trace=trace; self.writes=[]
    @property
    def ndim(self): return len(self.shape)
    def numel(self): return math.prod(self.shape)
    def stride(self): return self._stride
    def tolist(self): return list(self.values)
    def derive(self, shape, operation, stride=None):
        return Tensor(shape,self.dtype,self.name,stride,trace=self.trace+(operation,))
    def __getitem__(self, index):
        if isinstance(index,int): return self.derive(self.shape[1:],('index',index),self._stride[1:])
        start,stop,step=index.indices(self.shape[0]); assert step==1
        return self.derive((stop-start,)+self.shape[1:],('slice',start,stop),self._stride)
    def __setitem__(self,index,value):
        start,stop,step=index.indices(self.shape[0]); assert step==1
        assert value.shape==(stop-start,)+self.shape[1:]
        self.writes.append((start,stop,value))
    def contiguous(self): return self.derive(self.shape,('contiguous',))
    def as_strided(self,shape,stride): return self.derive(shape,('as_strided',tuple(shape),tuple(stride)),tuple(stride))
    def permute(self,dims): return self.derive(tuple(self.shape[i] for i in dims),('permute',tuple(dims)),tuple(self._stride[i] for i in dims))
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],tuple): shape=shape[0]
        shape=list(shape)
        if -1 in shape: shape[shape.index(-1)]=self.numel()//math.prod(n for n in shape if n!=-1)
        assert math.prod(shape)==self.numel()
        return self.derive(shape,('reshape',tuple(shape)))
    @property
    def T(self): assert self.ndim==2; return self.permute((1,0))
    def movedim(self,src,dst):
        dims=list(range(self.ndim)); v=dims.pop(src); dims.insert(dst,v); return self.permute(dims)

def pad(x,padding):
    return x.derive(x.shape[:-1]+(x.shape[-1]+sum(padding),),('pad',padding))
api=SimpleNamespace(bfloat16='bfloat16',int32='int32',
    empty=lambda shape,**kw: Tensor(shape,kw['dtype'],'output'),
    nn=SimpleNamespace(functional=SimpleNamespace(pad=pad)))
class Native:
    """Only records the existing factory protocol; never multiplies operands."""
    def __init__(self): self.calls=[]; self.factories=[]
    def factory(self,case):
        op=case._call.signature.name; self.factories.append(op)
        def run(a,b,bias=None):
            self.calls.append((a,b,bias))
            if op=='BmmFwdOp':
                assert a.ndim==b.ndim==3 and a.shape[0]==b.shape[0]
                shape=(a.shape[0],a.shape[1],b.shape[2])
            else:
                assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[0]
                shape=(a.shape[0],b.shape[1])
            return Tensor(shape,a.dtype,'native_mock',trace=(('native_result',len(self.calls)-1),))
        return SimpleNamespace(implementation=SimpleNamespace(run=run),cleanup=lambda:None)

scope=json.loads((D/'SCOPE.json').read_text())
keys={(s['baseline'],s['operator'],s['case_id']) for s in scope}
mapping=json.loads((R/'bf16_other_contract20/SOURCE_MAPPING.json').read_text())
rows=[r for r in mapping['rows'] if tuple(r['key']) in keys]
assert len(rows)==len(keys)==10
specfiles={'BmmFwdOp':'gemm_bmm','Conv1dFwdOp':'convolution','GroupedGemmFwdOp':'gemm'}
specs={op:yaml.safe_load((W/f'TileOPs/src/tileops/manifest/spec/{f}.yaml').read_text())[op] for op,f in specfiles.items()}
for row in rows:
    op,cid=row['key'][1:]; spec=specs[op]
    matches=[]
    for workload in spec['workloads']:
        for dt in workload.get('dtype_cases',[]):
            # Exact expanded key from manifest label and explicit dtype case.
            if workload['label']+'-'+dt['T']==cid:
                p={k:v['default'] for k,v in spec['signature'].get('params',{}).items() if 'default' in v}
                p.update({k:v for k,v in workload.items() if k not in ('label','dtype_cases','some')});p.update(dt)
                if op=='Conv1dFwdOp': p['bias']='bias' in workload.get('some',[])
                matches.append(p)
    assert len(matches)==1
    row['manifest_params']=matches[0]
    row['manifest_source']=f'TileOPs/src/tileops/manifest/spec/{specfiles[op]}.yaml'
    for k,v in row['params'].items():
        if k=='dtype': assert v==matches[0]['T']
        else: assert matches[0][k]==v,(k,v,matches[0])

def inputs(row):
    p=row['manifest_params'];op=row['key'][1]
    if op=='BmmFwdOp': return (Tensor((p['B'],p['M'],p['K']),name='a'),Tensor((p['B'],p['K'],p['N']),name='b'))
    if op=='Conv1dFwdOp': return (Tensor((p['N'],p['C_in'],p['L_in']),name='x'),Tensor((p['C_out'],p['C_in'],p['kW']),name='w'),Tensor((p['C_out'],),name='bias'))
    return (Tensor((p['M'],p['K']),name='a'),Tensor((p['G'],p['N'],p['K']),name='b'),Tensor((p['G'],),'int32','sizes',values=row['metadata']['sizes']),Tensor((p['G'],),'int32','offsets',values=row['metadata']['offsets']))

for row in rows:
    def exercise(row=row):
        n=Native(); op=row['key'][1];args=inputs(row)
        run,_=adapter.make_runner(op,row['manifest_params'],native_factory=n.factory,tensor_api=api)
        out=run(*args)
        assert out.shape==tuple(row['output']['shape']) and out.dtype==row['manifest_params']['T']
        assert all(not a.writes for a in args)  # Inputs and metadata not scattered into.
        if op=='BmmFwdOp':
            assert n.factories==['BmmFwdOp'] and len(n.calls)==1
            assert n.calls[0][:2]==args
        elif op=='Conv1dFwdOp':
            assert n.factories==['GemmFwdOp'] and len(n.calls)==1
            lhs,rhs,bias=n.calls[0]
            assert lhs.shape==(2998,240) and rhs.shape==(240,1280) and bias is args[2]
            assert ('as_strided',(1,80,2998,3),(240000,3000,1,1)) in lhs.trace
            assert ('permute',(0,2,1,3)) in lhs.trace
            assert rhs.trace[:2]==(('reshape',(1280,240)),('permute',(1,0)))
            assert out.stride()==(3837440,2998,1)
            rejects(lambda:run(args[0],args[1],None))
        else:
            live=[(i,start,size) for i,(start,size) in enumerate(zip(row['metadata']['offsets'],row['metadata']['sizes'])) if size]
            assert len(n.calls)==len(live)==len(out.writes)
            for (i,start,size),(lhs,rhs,bias),(lo,hi,y) in zip(live,n.calls,out.writes):
                assert ('slice',start,start+size) in lhs.trace
                assert ('index',i) in rhs.trace and ('permute',(1,0)) in rhs.trace
                assert (lo,hi)==(start,start+size) and y.trace==(('native_result',live.index((i,start,size))),)
            assert sum(hi-lo for lo,hi,_ in out.writes)==row['manifest_params']['M']
        row['cpu_mock']={'status':'pass','factory_modes':n.factories,'product_calls':len(n.calls),'output_shape':list(out.shape),'output_dtype':out.dtype,'numeric_validation':False}
    record('/'.join(row['key']),exercise)

empty=next(r for r in rows if 'empty' in r['key'][2]);p=empty['manifest_params'];a,b,s,o=inputs(empty)
def grouped_edgecases():
    n=Native();run,_=adapter.make_runner('GroupedGemmFwdOp',p,native_factory=n.factory,tensor_api=api)
    for bad in ([0]*16,[-1]+o.values[1:],o.values[:-1]+[9]):
        rejects(lambda:run(a,b,s,Tensor((16,),'int32',values=bad)))
    rejects(lambda:run(a,b,Tensor((16,),'int32',values=[1]*16),o))
    rejects(lambda:run(a,b,Tensor((16,),'int32',values=[-1]+[0]*15),o))
    assert not n.calls
    rejects(lambda:adapter.make_runner('GroupedGemmFwdOp',p|{'transpose_a':True},native_factory=n.factory,tensor_api=api),NotImplementedError)
    assert len(n.factories)==1
    # NN with eight live + eight empty groups, and interspersed empties retain weight positions.
    for ns in ([1]*8+[0]*8,[0,1]*8):
        nn=Native();run,_=adapter.make_runner('GroupedGemmFwdOp',p|{'transpose_b':False},native_factory=nn.factory,tensor_api=api)
        starts=[];end=0
        for size in ns: starts.append(end);end+=size
        out=run(a,Tensor((16,p['K'],p['N']),name='b'),Tensor((16,),'int32',values=ns),Tensor((16,),'int32',values=starts))
        assert len(nn.calls)==8 and out.shape==(8,p['N'])
        assert [rhs.trace[0][1] for _,rhs,_ in nn.calls]==[i for i,size in enumerate(ns) if size]
        assert [(lo,hi) for lo,hi,_ in out.writes]==[(i,i+1) for i in range(8)]
    # All-empty interface contract: genuinely zero rows, no placeholder native call.
    nn=Native();run,_=adapter.make_runner('GroupedGemmFwdOp',p|{'M':0},native_factory=nn.factory,tensor_api=api)
    z=Tensor((16,),'int32',values=[0]*16)
    out=run(Tensor((0,p['K'])),b,z,z)
    assert out.shape==(0,p['N']) and not nn.calls and not out.writes
record('grouped_empty_NN_NT_offsets_rejections_and_all_empty',grouped_edgecases)

def dtype_checks():
    n=Native();run,_=adapter.make_runner('GroupedGemmFwdOp',p,native_factory=n.factory,tensor_api=api)
    rejects(lambda:run(Tensor(a.shape,'float16'),b,s,o)); assert not n.calls
    def bad_factory(case):
        return SimpleNamespace(implementation=SimpleNamespace(run=lambda a,b: Tensor((a.shape[0],b.shape[1]),'float32')))
    run,_=adapter.make_runner('GroupedGemmFwdOp',p,native_factory=bad_factory,tensor_api=api)
    rejects(lambda:run(a,b,s,o))
    rejects(lambda:adapter.make_runner('GroupedGemmFwdOp',p|{'T':'float16'},native_factory=n.factory,tensor_api=api))
record('input_output_dtype_contract_rejections',dtype_checks)

def loading():
    before=dict(sys.modules)
    rejects(lambda:adapter.load_factory(D/'adapter.py',sha256='0'*64))
    # Load the actual adapter as harmless source to exercise isolated module construction.
    factory=adapter.load_factory(D/'adapter.py',sha256=hashlib.sha256((D/'adapter.py').read_bytes()).hexdigest(),symbol='make_runner')
    assert callable(factory) and dict(sys.modules)==before
    rejects(lambda:adapter.make_runner('GroupedGemmFwdOp',p,native_factory=None,tensor_api=api),TypeError)
record('explicit_pin_no_fallback_no_sysmodules_alias',loading)
def original_v5_argument_mock():
    # Execute the existing Python binding with explicitly intercepted imports.
    # All tensor/ACL operations below are metadata stubs; no library is loaded.
    import builtins
    import ctypes
    events=[]; descriptors={}; tensorlists={}; serial=[100]
    def pointer(): serial[0]+=1; return serial[0]
    class Function:
        def __init__(self, fn): self.fn=fn
        def __call__(self,*args): return self.fn(*args)
    def create_list(arr,count):
        ptr=pointer(); tensorlists[ptr]=[descriptors[arr[i]] for i in range(count)]
        events.append(('create',ptr)); return ptr
    def destroy_list(ptr): events.append(('destroy',ptr)); return 0
    class MathCall:
        def __init__(self,name,argtypes,**kwargs):
            assert name=='aclnnGroupedMatmulV5' and len(argtypes)==20
            self.base=SimpleNamespace(aclCreateTensorList=Function(create_list),aclDestroyTensorList=Function(destroy_list))
        def record(self): return {'keep':[],'tensors':[]}
        def tensor(self,t,record):
            ptr=pointer();descriptors[ptr]=t;record['tensors'].append(ptr);return ptr
        def call(self,args,record,device):
            assert len(args)==20 and args[3:12]==[None]*9
            assert args[12:17]==[0,-1,0,0,None] and args[18:]==[None,None]
            assert not record['tensors'] and device=='mock'
            xs,ws,ys=(tensorlists[args[i]] for i in (0,1,17))
            assert len(xs)==len(ws)==len(ys)
            for x,w,y in zip(xs,ws,ys):
                assert x.dtype==w.dtype==y.dtype=='float32'
                assert x.shape[1]==w.shape[0] and y.shape==(x.shape[0],w.shape[1])
            if args[2] is not None:
                bias=tensorlists[args[2]]; assert len(bias)==1 and bias[0].dtype=='float32'
            events.append(('call',args,xs,ws,ys))
        def check(self,code,label): assert code==0
        def cleanup(self): events.append(('cleanup',))
    def cast(self,dtype):
        out=self.derive(self.shape,('cast',dtype));out.dtype=dtype;return out
    Tensor.float=lambda self:cast(self,'float32')
    Tensor.to=cast
    Tensor.__iter__=lambda self:iter([self[i] for i in range(self.shape[0])])
    torch_mock=SimpleNamespace(float32='float32',empty=lambda shape,**kw:Tensor(shape,kw['dtype']),
        stack=lambda xs:Tensor((len(xs),)+xs[0].shape,xs[0].dtype),
        npu=SimpleNamespace(synchronize=lambda:events.append(('synchronize',))))
    imports={
        'torch':torch_mock,
        'benchmarks.ascend_math_more':SimpleNamespace(_MathCall=MathCall),
        'benchmarks.ascend_baselines':SimpleNamespace(Candidate=lambda impl,*args,**kwargs:SimpleNamespace(implementation=impl,cleanup=kwargs['cleanup']),source_hashes=lambda *files:{}),
        'benchmarks.api':SimpleNamespace(Implementation=lambda run:SimpleNamespace(run=run))}
    def intercepted(name,*args,**kwargs):
        if name in imports: return imports[name]
        if name in ('pathlib','ctypes'): return builtins.__import__(name,*args,**kwargs)
        raise AssertionError('Unexpected import: '+name)
    source=R/'gmm_fp16_remaining11/v5_fp32_bias.py'
    env={'__file__':str(source),'__builtins__':dict(vars(builtins),__import__=intercepted)}
    exec(compile(source.read_bytes(),str(source),'exec'),env)
    for row in rows:
        op=row['key'][1];p=row['manifest_params']
        run,inner=adapter.make_runner(op,p,native_factory=env['dense_fp32'],tensor_api=api)
        before=len([e for e in events if e[0]=='call'])
        out=run(*inputs(row));assert out.dtype=='bfloat16'
        calls=[e for e in events if e[0]=='call'][before:]
        if op=='BmmFwdOp':
            assert len(calls)==1 and len(calls[0][2])==p['B']
            assert [x.trace[1] for x in calls[0][2]]==[('index',i) for i in range(p['B'])]
        elif op=='Conv1dFwdOp': assert len(calls)==1 and calls[0][1][2] is not None
        else: assert len(calls)==sum(n>0 for n in row['metadata']['sizes'])
        inner.cleanup()
        cleanup_index=len(events)-1
        sync_index=max(i for i,e in enumerate(events) if e[0]=='synchronize')
        assert events[cleanup_index][0]=='cleanup'
        assert all(e[0]=='destroy' for e in events[sync_index+1:cleanup_index])
    assert len([e for e in events if e[0]=='create'])==len([e for e in events if e[0]=='destroy'])
record('original_complete_V5_TensorList_argument_order_widen_cast_lifetime_mock_all10',original_v5_argument_mock)
for name in ('adapter.py','cpu_checks.py'): ast.parse((D/name).read_text())
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
(D/'CONTRACTS.json').write_text(json.dumps({'rows':rows,'native_contract':mapping['common_native_contract'],'params_source':'Exact scoped YAML workload label + explicit dtype_cases.T and signature defaults; never filename dtype inference'},indent=2)+'\n')
(D/'CPU_CHECKS.json').write_text(json.dumps({'status':'pass','checks':checks,'count':len(checks),'torch_imported':False,'native_loaded':False,'NPU_runs':0,'reference_runs':0,'numerical_validation':False,'limits':'Pure Python metadata mock verifies wrapper routing/shape/strides/bias identity/scatter/prefix/dtype rejection only. Original V5 Python control flow is additionally executed with intercepted fake torch/MathCall imports to verify all 20 API argument positions, batch lists, cast order and synchronize-before-destroy. No real torch operators, ACL/native ABI, arithmetic, accumulation, or canonical checker execute.'},indent=2)+'\n')
print(json.dumps({'checks_passed':len(checks),'scope_keys':len(rows),'numeric_validation':False}))
