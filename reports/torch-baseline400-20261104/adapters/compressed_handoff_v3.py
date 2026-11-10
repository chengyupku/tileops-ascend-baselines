"""Lossless CPU-storage chunks, bounded SHM/disk use, preserving tensor views/aliases."""
import pathlib,hashlib,shutil,fcntl,gzip,time

def sha(f):
 with open(f,'rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()

def save(bundle,root,tid,timeout=120):
 import torch
 root=pathlib.Path(root);dirs=[pathlib.Path('/dev/shm/torch-baseline400'),root/'spool']
 for d in dirs:d.mkdir(exist_ok=True)
 stores={};specs=[];ids={}
 def encode(x):
  if isinstance(x,torch.Tensor):
   assert x.device.type=='cpu' and x.layout==torch.strided
   if id(x) not in ids:
    s=x.untyped_storage();key=(s.data_ptr(),s.nbytes());store=stores.setdefault(key,{'index':len(stores),'storage':s})
    ids[id(x)]=len(specs);specs.append({'storage':store['index'],'dtype':str(x.dtype).split('.')[-1],'shape':list(x.shape),'stride':list(x.stride()),'offset':x.storage_offset()})
   return ['tensor',ids[id(x)]]
  if isinstance(x,dict):return ['dict',[(k,encode(v)) for k,v in x.items()]]
  if isinstance(x,tuple):return ['tuple',[encode(v) for v in x]]
  if isinstance(x,list):return ['list',[encode(v) for v in x]]
  return ['value',x]
 tree=encode(bundle);parts=[];chunk=256*1024**2
 try:
  for store in stores.values():
   s=store['storage'];raw=torch.empty(0,dtype=torch.uint8).set_(s,0,(s.nbytes(),),(1,));view=memoryview(raw.numpy());files=[]
   for offset in range(0,s.nbytes(),chunk):
    size=min(chunk,s.nbytes()-offset);deadline=time.monotonic()+timeout
    while True:
     with (root/'handoff-capacity.lock').open('a') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX)
      available=[shutil.disk_usage(d).free-reserve for d,reserve in zip(dirs,[1024**3,52*1024**3])]
      choice=next((i for i,free in enumerate(available) if free>size+64*1024**2),None)
      if choice is not None:
       f=dirs[choice]/f'{tid}.shard{store["index"]}-{offset}.gz';parts.append(f)
       with gzip.open(f,'wb',compresslevel=1) as z:z.write(view[offset:offset+size])
       files.append({'path':str(f),'offset':offset,'raw_bytes':size,'size':f.stat().st_size,'sha256':sha(f)});break
     if time.monotonic()>deadline:raise RuntimeError('Compressed CPU handoff capacity unavailable; 52GiB disk reserve preserved')
     time.sleep(1)
   store['files']=files;store['raw_bytes']=s.nbytes()
  return {'format':'compressed-storage-chunks-v3','tree':tree,'tensors':specs,'storages':[{k:v for k,v in s.items() if k!='storage'} for s in stores.values()]}
 except BaseException:
  for f in parts:f.unlink(missing_ok=True)
  raise

def load(payload):
 import torch
 stores=[]
 for s in payload['storages']:
  raw=torch.empty(s['raw_bytes'],dtype=torch.uint8);view=memoryview(raw.numpy())
  for f in s['files']:
   path=pathlib.Path(f['path']);assert path.stat().st_size==f['size'] and sha(path)==f['sha256']
   with gzip.open(path,'rb') as z:
    target=view[f['offset']:f['offset']+f['raw_bytes']];n=z.readinto(target);assert n==len(target) and not z.read(1)
   path.unlink()
  stores.append(raw.untyped_storage())
 values=[torch.empty(0,dtype=getattr(torch,s['dtype'])).set_(stores[s['storage']],s['offset'],s['shape'],s['stride']) for s in payload['tensors']]
 def decode(x):
  k,v=x
  if k=='tensor':return values[v]
  if k=='dict':return {a:decode(b) for a,b in v}
  if k=='list':return [decode(a) for a in v]
  if k=='tuple':return tuple(decode(a) for a in v)
  assert k=='value';return v
 return decode(payload['tree'])
