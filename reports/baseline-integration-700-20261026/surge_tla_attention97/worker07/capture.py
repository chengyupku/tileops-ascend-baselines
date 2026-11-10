from pathlib import Path
import json,hashlib,ctypes,inspect,shutil,os

def identity(p):
 p=Path(p);b=p.read_bytes();s=p.stat()
 return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),inode=s.st_ino,device=s.st_dev,elf_machine=int.from_bytes(b[18:20],'little') if b[:4]==b'\x7fELF' else None)
def snapshot(out,stage):
 maps=Path('/proc/self/maps').read_text();(out/(stage+'_maps.txt')).write_text(maps);d={}
 for line in maps.splitlines():
  a=line.split(None,5)
  if len(a)==6 and a[-1].startswith('/') and any(n in a[-1] for n in ['tilelang','libtvm','libascend','libruntime','torch_npu','tmp']):
   p=Path(a[-1])
   if p.is_file() and str(p) not in d:d[str(p)]=identity(p)
 (out/(stage+'_loaded.json')).write_text(json.dumps(d,indent=2))
def install(out,tl,torch):
 from tilelang.jit.kernel import JITKernel
 old=JITKernel.__call__;count=[0]
 def call(k,*args,**kwargs):
  count[0]+=1;d=out/('launch_%02d'%count[0]);d.mkdir()
  a=k.adapter;p=Path(a.libpath);r=dict(binary=identity(p),target=str(k.target),execution_backend=k.execution_backend,adapter_class=str(type(a)),functions={})
  lib=getattr(a,'lib',None)
  for name in ['call','init','get_last_error']:
   f=getattr(lib,name,None)
   if f is not None:r['functions'][name]=ctypes.cast(f,ctypes.c_void_p).value
  for name,v in vars(a).items():
   if isinstance(v,ctypes._CFuncPtr):r['functions'][name]=ctypes.cast(v,ctypes.c_void_p).value
  (d/'native.json').write_text(json.dumps(r,indent=2));shutil.copy2(p,d/p.name)
  (d/'kernel.txt').write_text(k.get_kernel_source());snapshot(d,'before_launch')
  return old(k,*args,**kwargs)
 JITKernel.__call__=call
 root=Path(tl.__file__).parent.parent
 files=[root/'tilelang/jit/adapter/libgen.py',root/'tilelang/engine/phase.py',root/'tilelang/utils/target.py',root/'3rdparty/pto-isa/include/pto/pto-inst.hpp']
 sdk=Path(os.environ.get('ASCEND_HOME_PATH','/usr/local/Ascend/ascend-toolkit/latest'))
 files += [sdk/'include/acl/acl.h',sdk/'include/kernel_operator.h']
 b=shutil.which('bisheng')
 if b:files.append(Path(b).resolve())
 (out/'build_inputs.json').write_text(json.dumps({'files':[identity(p) for p in files if p.is_file()],'sdk':str(sdk),'platform':'A5','target':'original decorator auto for GQA; original AscendC for elementwise','explicit_passes':'unchanged original decorator'},indent=2))
