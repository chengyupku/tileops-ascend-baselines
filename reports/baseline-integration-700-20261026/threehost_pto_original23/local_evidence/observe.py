from pathlib import Path
import os,sys,json,hashlib,ctypes,time,shutil

def install(out,tl,torch):
 def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
 def identity(f):
  f=Path(f);s=f.stat();return dict(path=str(f),realpath=str(f.resolve()),sha256=sha(f),inode=s.st_ino,device=s.st_dev,size=s.st_size)
 def loaded():
  maps=Path('/proc/self/maps').read_text();result=[]
  for p in sorted({l.split()[-1] for l in maps.splitlines() if len(l.split())>5 and l.split()[-1].startswith('/') and any(n in l for n in ['libtilelang','libtvm','libascendcl','libruntime.so','libtorch_npu'])}):
   if Path(p).is_file():result.append(identity(p))
  return dict(maps=maps,libraries=result,python_modules=[identity(m.__file__) for n,m in list(sys.modules.items()) if n.startswith(('tilelang','tvm')) and getattr(m,'__file__',None) and Path(m.__file__).is_file()])
 (out/'loaded_before_compile.json').write_text(json.dumps(loaded(),indent=2))
 def audit(event,args):
  if event=='subprocess.Popen':
   with (out/'subprocess.jsonl').open('a') as f:f.write(json.dumps(dict(event=event,executable=args[0],argv=args[1],cwd=args[2]),default=str)+'\n')
 sys.addaudithook(audit)
 class Dl_info(ctypes.Structure):_fields_=[('dli_fname',ctypes.c_char_p),('dli_fbase',ctypes.c_void_p),('dli_sname',ctypes.c_char_p),('dli_saddr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl_info)];dl.dladdr.restype=ctypes.c_int
 from tilelang.jit.kernel import JITKernel
 original=JITKernel.__call__;seq=0
 def trace(k,*args,**kwargs):
  nonlocal seq
  (out/'DEVICE.json').write_text(json.dumps(dict(device_name=torch.npu.get_device_name(0)),indent=2))
  seq+=1;ad=k.adapter;lib=ad.lib;f=Path(lib._name);r=identity(f);dest=out/'native_artifacts'/r['sha256'];dest.parent.mkdir(exist_ok=True);shutil.copyfile(f,dest)
  r.update(sequence=seq,target=str(k.target),platform=str(getattr(k,'platform',getattr(ad,'platform',None))),pass_configs=str(getattr(k,'pass_configs',None)),symbols={},loaded=loaded())
  for name in ['call']:
   fun=getattr(lib,name);ptr=ctypes.cast(fun,ctypes.c_void_p).value;info=Dl_info();rc=dl.dladdr(ptr,ctypes.byref(info));r['symbols'][name]=dict(pointer=ptr,dladdr_rc=rc,filename=info.dli_fname.decode() if info.dli_fname else None,base=info.dli_fbase,symbol=info.dli_sname.decode() if info.dli_sname else None,symbol_address=info.dli_saddr)
  (out/f'{seq}.kernel.txt').write_text(k.get_kernel_source())
  gen=getattr(ad,'lib_generator',None)
  r['libgen']=dict(target=getattr(gen,'target',None),platform=getattr(gen,'platform',None),compile_flags=getattr(gen,'compile_flags',None),source_path=str(getattr(gen,'srcpath',None)))
  src=getattr(gen,'srcpath',None)
  if src and Path(src).exists():shutil.copyfile(src,out/f'{seq}.generated.cpp')
  path=out/f'{seq}.dispatch.json';path.write_text(json.dumps(r,indent=2));result=original(k,*args,**kwargs);torch.npu.synchronize();r['completed']=time.time();path.write_text(json.dumps(r,indent=2));return result
 JITKernel.__call__=trace
