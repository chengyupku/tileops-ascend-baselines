"""Lossless CPU tensor-storage shards across SHM/disk; no device imports."""
import pathlib,hashlib,shutil,fcntl,time

def sha(p):
 with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def placement(sizes, capacities, overhead=64*1024**2):
 remaining=list(capacities);result={}
 for key,size in sorted(sizes.items(),key=lambda x:-x[1]):
  choices=[i for i,free in enumerate(remaining) if free>=size+overhead]
  if not choices:return None
  i=choices[0];remaining[i]-=size+overhead;result[key]=i
 return result

def save(bundle, root, tid, timeout=600):
 import torch
 root=pathlib.Path(root);dirs=[pathlib.Path('/dev/shm/torch-baseline1135'),root/'spool']
 for d in dirs:d.mkdir(exist_ok=True)
 tensors=[];ids={};groups={};sizes={}
 def encode(x):
  if isinstance(x,torch.Tensor):
   assert x.device.type=='cpu'
   if id(x) not in ids:
    i=len(tensors);ids[id(x)]=i;tensors.append(x);s=x.untyped_storage();key=(str(x.device),s.data_ptr(),s.nbytes());groups.setdefault(key,[]).append(i);sizes[key]=s.nbytes()
   return ('tensor',ids[id(x)])
  if isinstance(x,dict):return ('dict',[(k,encode(v)) for k,v in x.items()])
  if isinstance(x,tuple):return ('tuple',[encode(v) for v in x])
  if isinstance(x,list):return ('list',[encode(v) for v in x])
  return ('value',x)
 tree=encode(bundle);deadline=time.monotonic()+timeout
 while True:
  with (root/'handoff-capacity.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   plan=placement(sizes,[shutil.disk_usage(dirs[0]).free-1024**3,shutil.disk_usage(dirs[1]).free-52*1024**3])
   if plan is not None:
    paths=[];files=[]
    try:
     for n,(key,location) in enumerate(plan.items()):
      f=dirs[location]/(tid+'.shard'+str(n)+'.pt');partial=f.with_suffix('.partial');paths.extend([f,partial]);indices=groups[key]
      torch.save([tensors[i] for i in indices],partial);partial.replace(f)
      files.append({'path':str(f),'size':f.stat().st_size,'sha256':sha(f),'indices':indices})
     return {'format':'tensor-storage-shards-v1','tree':tree,'files':files,'tensor_count':len(tensors)}
    except BaseException:
     for f in paths:f.unlink(missing_ok=True)
     raise
  if time.monotonic()>=deadline:raise RuntimeError('SHM+disk shard capacity unavailable; reserves unchanged')
  time.sleep(1)

def load(payload):
 import torch
 values=[None]*payload['tensor_count']
 for f in payload['files']:
  path=pathlib.Path(f['path']);assert path.stat().st_size==f['size'] and sha(path)==f['sha256']
  xs=torch.load(path,map_location='cpu',weights_only=False);assert len(xs)==len(f['indices'])
  for i,x in zip(f['indices'],xs):values[i]=x
  path.unlink()
 def decode(node):
  kind,value=node
  if kind=='tensor':return values[value]
  if kind=='dict':return {k:decode(v) for k,v in value}
  if kind=='tuple':return tuple(decode(v) for v in value)
  if kind=='list':return [decode(v) for v in value]
  assert kind=='value';return value
 return decode(payload['tree'])
