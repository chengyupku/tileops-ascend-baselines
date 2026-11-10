"""Phase1 only; run one PLAN id through run_probe.py. Never reads storage or allocates workspace."""
import argparse,ctypes as C,hashlib,json,math,os,time,traceback
from pathlib import Path
from resource_gate import capacity
H=Path(__file__).resolve().parent
OWNER='baseline700-20261026/moe_down_checker_query3'
p=argparse.ArgumentParser();p.add_argument('id');p.add_argument('--device-query',action='store_true');p.add_argument('--attempt',default='');a=p.parse_args()
s=next(x for x in json.loads((H/'PLAN.json').read_text()) if x['id']==a.id)
assert s['payload_bytes']<=20*2**30
if not a.device_query: print(json.dumps(s));raise SystemExit
R=H/(a.id+a.attempt);R.mkdir(exist_ok=True)
report={'spec':s,'kernel_launches':0,'workspace_allocations':0,'malloc_policy':'ACL_MEM_MALLOC_HUGE_FIRST (0)','events':[],'queries':[]}
def save(): (R/'result.json').write_text(json.dumps(report,indent=2))
def event(name,**kw): report['events'].append(dict(name=name,time=time.time(),**kw));save()
def checked(rc,name):
 event(name,rc=rc)
 if rc: raise RuntimeError(f'{name}: {rc}')
def identity():
 m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert m['owner']==OWNER and m['phase']=='running';assert m['child']['pid']==os.getpid();assert m['child']['start_ticks']==int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]);assert m['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 fd=[str(f) for f in Path('/proc/self/fd').iterdir() if f.exists() and os.stat(f).st_ino==os.stat('/run/lock/ascend-npu/device-0.lock').st_ino and os.stat(f).st_dev==os.stat('/run/lock/ascend-npu/device-0.lock').st_dev];assert fd
 return {'metadata':m,'inherited_lock_fds':fd,'pid':os.getpid(),'pgid':os.getpgrp()}
report['liveguard']=identity();report['host_preinit']=capacity();save();assert report['host_preinit']['effective_available_host_bytes']>=32*2**30
V,U,I,P=C.c_void_p,C.c_uint64,C.c_int64,C.POINTER
sdk=Path('/usr/local/Ascend/cann-9.2.0-beta.2');rt=C.CDLL(str(sdk/'lib64/libascendcl.so'),mode=C.RTLD_GLOBAL);base=C.CDLL(str(sdk/'lib64/libnnopbase.so'),mode=C.RTLD_GLOBAL);api=C.CDLL(str(sdk/'lib64/libopapi.so'),mode=C.RTLD_GLOBAL)
def bind(lib,n,args,ret=C.c_int):
 f=getattr(lib,n);f.argtypes=args;f.restype=ret;return f
init=bind(rt,'aclInit',[C.c_char_p]);device=bind(rt,'aclrtSetDevice',[C.c_int]);malloc=bind(rt,'aclrtMalloc',[P(V),C.c_size_t,C.c_int]);free=bind(rt,'aclrtFree',[V]);sync=bind(rt,'aclrtSynchronizeDevice',[]);fin=bind(rt,'aclFinalize',[]);mem=bind(rt,'aclrtGetMemInfo',[C.c_int,P(C.c_size_t),P(C.c_size_t)])
create=bind(base,'aclCreateTensor',[P(I),U,C.c_int,P(I),I,C.c_int,P(I),U,V],V);destroy=bind(base,'aclDestroyTensor',[V]);array=bind(base,'aclCreateIntArray',[P(I),U],V);da=bind(base,'aclDestroyIntArray',[V]);de=bind(base,'aclDestroyAclOpExecutor',[V])
close=bind(api,'aclnnIsCloseGetWorkspaceSize',[V,V,C.c_double,C.c_double,C.c_bool,V,P(U),P(V)]);all_=bind(api,'aclnnAllGetWorkspaceSize',[V,V,C.c_bool,V,P(U),P(V)])
def provenance(f):
 addr=C.cast(f,V).value;lines=[l for l in Path('/proc/self/maps').read_text().splitlines() if int(l.split()[0].split('-')[0],16)<=addr<int(l.split()[0].split('-')[1],16)];assert len(lines)==1
 path=Path(lines[0].split(maxsplit=5)[5]);st=path.stat();return {'address':hex(addr),'procmap':lines[0],'path':str(path),'realpath':str(path.resolve()),'inode':st.st_ino,'device':st.st_dev,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
report['symbols']={n:provenance(f) for n,f in [('aclnnIsCloseGetWorkspaceSize',close),('aclnnAllGetWorkspaceSize',all_),('aclDestroyAclOpExecutor',de)]};save()
def memory():
 f,t=C.c_size_t(),C.c_size_t();checked(mem(1,C.byref(f),C.byref(t)),'aclrtGetMemInfo');return {'free':f.value,'total':t.value}
bufs=[];ts=[];exs=[];ars=[];keep=[];initialized=False;active=False
try:
 checked(init(None),'aclInit');initialized=True;checked(device(0),'aclrtSetDevice');active=True
 report['admission']={'host':capacity(),'hbm':memory(),'required_hbm_free':s['payload_bytes']+32*2**30,'payload_cap':20*2**30};save()
 assert report['admission']['host']['effective_available_host_bytes']>=32*2**30 and report['admission']['hbm']['free']>=report['admission']['required_hbm_free']
 def tensor(dims,code,width,fmt,label):
  n=math.prod(dims);b=V();checked(malloc(C.byref(b),n*width,0),'aclrtMalloc_'+label);assert b.value and b.value not in [x.value for x in bufs];bufs.append(b)
  strides=[math.prod(dims[i+1:]) for i in range(len(dims))];ds=(I*len(dims))(*dims);st=(I*len(dims))(*strides);storage=(I*1)(n);keep.extend([ds,st,storage]);t=create(ds,len(dims),code,st,0,fmt,storage,1,b);assert t;ts.append(t)
  event('descriptor',label=label,shape=dims,strides=strides,storage_shape=[n],format_code=fmt,dtype_code=code,device_ptr=hex(b.value),descriptor_ptr=hex(t),bytes=n*width);return t
 code,width={'bfloat16':(27,2),'int32':(3,4),'float32':(0,4)}[s['dtype']]
 x=tensor(s['shape'],code,width,s['format_code'],'self');y=tensor(s['shape'],code,width,s['format_code'],'other');matches=tensor(s['shape'],12,1,s['format_code'],'matches');scalar=tensor([],12,1,2,'scalar')
 d=(I*len(s['shape']))(*range(len(s['shape'])));keep.append(d);dims=array(d,len(s['shape']));assert dims;ars.append(dims)
 jobs=[('aclnnIsCloseGetWorkspaceSize',close,[x,y,s['rtol'],s['atol'],s['equal_nan'],matches])]
 if s['query_all']:jobs.append(('aclnnAllGetWorkspaceSize',all_,[matches,dims,False,scalar]))
 for name,f,args in jobs:
  report['liveguard_before_'+name]=identity();before=memory();ws,ex=U(),V();start=time.monotonic();rc=f(*args,C.byref(ws),C.byref(ex));elapsed=time.monotonic()-start
  if ex.value:exs.append(ex)
  report['queries'].append({'api':name,'rc':rc,'workspace_bytes':ws.value,'executor':hex(ex.value) if ex.value else None,'seconds':elapsed,'before_hbm':before,'after_hbm':memory()});save()
  # Failed phase1 retained as evidence; independent All metadata query can still proceed.
 report['status']='queried' if all(q['rc']==0 for q in report['queries']) else 'api_failure'
except BaseException as e:
 report['status']='failed';report['error']=repr(e);report['traceback']=traceback.format_exc();save()
finally:
 start=time.monotonic();cleanup=[]
 def clean(name,f,*args):
  t=time.monotonic();rc=f(*args);cleanup.append({'name':name,'rc':rc,'seconds':time.monotonic()-t});report['cleanup']=cleanup;save()
 if active: clean('aclrtSynchronizeDevice',sync)
 for ex in reversed(exs):clean('aclDestroyAclOpExecutor',de,ex)
 for t in reversed(ts):clean('aclDestroyTensor',destroy,t)
 for d in reversed(ars):clean('aclDestroyIntArray',da,d)
 for b in reversed(bufs):clean('aclrtFree',free,b)
 if active:report['after_cleanup_hbm']=memory()
 if initialized:clean('aclFinalize',fin)
 report['cleanup_seconds']=time.monotonic()-start;save()
print(json.dumps({'id':s['id'],'status':report['status'],'queries':report['queries'],'cleanup_seconds':report['cleanup_seconds']}),flush=True)
raise SystemExit(0 if report['status']=='queried' and all(c['rc']==0 for c in cleanup) else 1)
