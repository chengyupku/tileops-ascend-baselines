from pathlib import Path
import json,re,hashlib
P=Path(__file__).resolve().parent
for d in P.iterdir():
 if not d.is_dir() or not (d/'checker.json').exists():continue
 c=json.loads((d/'checker.json').read_text());api=c.get('api','').split('::')[-1].split(' ')[0]
 if not api:continue
 token=api.lower().replace('_','');records=[];kernels=set();vend=False;reg=False;launch=False
 for f in (d/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
  for i,l in enumerate(f.open(errors='replace')):
   if token in l.lower().replace('_','') and any(x in l for x in ['allpairs_nn_impl','classicpilot_nn_impl','kernelName','FunctionRegister','RegisterKernel','KernelLaunch','LaunchKernel','kernel_name','stub_func']):
    records.append(dict(file=str(f),line=i+1,text=l.strip()));vend|=('allpairs_nn_impl' in l or 'classicpilot_nn_impl' in l);reg|=('FunctionRegister' in l);launch|=('StreamLaunchKernelV1:' in l or 'StreamLaunchKernelV2:' in l)
    kernels.update(re.findall(r'te_[a-z0-9]+_[0-9a-f]{64}',l))
 hashes={}
 for k in kernels:
  for f in Path('/root/dyq/workspace/.cache/all-pairs/ops_nn/graph_cache').rglob(k+'.o'):hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 (d/'ATTRIBUTION.json').write_text(json.dumps(dict(api=api,vendor_seen=vend,registration_seen=reg,device_launch_seen=launch,kernel_hashes=hashes,records=records),indent=2))
 print(d.name,c.get('status'),vend,reg,len(hashes))
