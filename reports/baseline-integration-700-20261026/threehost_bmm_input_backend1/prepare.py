from control import *
import ast,hashlib
hb('prepare_additive_capture')
old=P.parent/'threehost_bmm1'
for name in ['adapter.py','binding.py','native_observation.py','resource_gate.py','CASES.json']:(P/name).write_bytes((old/name).read_bytes())
s=(old/'capture.py').read_text().replace(OLD,R)
a=s.index(" if 'input' in name:");b=s.index(' v=x.detach()',a)
s=s[:a]+s[b:]
s=s.replace("if store:(P/(name+'.bin')).write_bytes(raw)","""if store:
  digest=hashlib.sha256(raw).hexdigest()
  previous=next((k for k,v in captures.items() if v.get('sha256')==digest and v.get('stored')),None)
  if previous:os.link(P/(previous+'.bin'),P/(name+'.bin'))
  else:(P/(name+'.bin')).write_bytes(raw)""")
s=s.replace("'stored':store}","'stored':store,'byteorder':sys.byteorder,'original_stride':list(x.stride()),'storage_offset':x.storage_offset(),'serialization':'contiguous logical tensor uint8 bytes'}")
insert='''
def settings():
 result={'environment':{k:v for k,v in os.environ.items() if k.startswith(('DNNL','ONEDNN','OMP','MKL','TORCH'))},'torch_version':torch.__version__,'git_version':torch.version.git_version,'num_threads':torch.get_num_threads(),'num_interop_threads':torch.get_num_interop_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
 for path in ['get_float32_matmul_precision','backends.mkldnn.enabled','backends.mkldnn.deterministic','backends.mkldnn.allow_tf32','backends.mkldnn.matmul.fp32_precision','backends.fp32_precision','backends.cuda.matmul.allow_bf16_reduced_precision_reduction']:
  try:
   obj=torch
   for part in path.split('.'):obj=getattr(obj,part)
   result[path]=obj() if callable(obj) else obj
  except Exception as e:result[path]={'unavailable':str(e)}
 return result
original_cpu_reference=c.cpu_reference
def instrumented_cpu_reference(torch_arg,reference,inputs):
 @functools.wraps(reference)
 def record(*args):
  dump('CPU_REFERENCE_SETTINGS.json',settings())
  for i,x in enumerate(args):save(f'cpu_reference_input{i}',x)
  print('CPU_REFERENCE_BEGIN',flush=True)
  output=reference(*args)
  print('CPU_REFERENCE_END',flush=True)
  return output
 return original_cpu_reference(torch_arg,record,inputs)
c.cpu_reference=instrumented_cpu_reference
'''
s=s.replace('original_observe=c.observe_call;observe_count=0',insert+'\noriginal_observe=c.observe_call;observe_count=0')
s=s.replace("rc=c.main()","rc=c.main()\nassert json.loads((P/'representative.json').read_text())['input_seed']==1868357590")
(P/'capture.py').write_text(s)
run=(old/'run.sh').read_text().replace(OLD,R).replace('export ASCEND_GLOBAL_LOG_LEVEL=0','export DNNL_VERBOSE=1\nexport ASCEND_GLOBAL_LOG_LEVEL=0')
(P/'run.sh').write_text(run)
for f in P.glob('*.py'):ast.parse(f.read_text())
setup=f'''from pathlib import Path
import shutil,json,hashlib,platform
old=Path({OLD!r});p=Path({R!r});p.mkdir(exist_ok=False)
for name in ['TileOPs','tools','private_provider']:shutil.copytree(old/name,p/name,ignore=shutil.ignore_patterns('__pycache__'))
print('copied original closure')
'''
r=py(setup);(P/'COPY_RECEIPT.json').write_text(json.dumps(r,indent=2));assert r['rc']==0,r
for name in ['adapter.py','binding.py','native_observation.py','resource_gate.py','CASES.json','capture.py','run.sh']:upload(name)
pre=f'''from pathlib import Path
import sys,json,hashlib,ast,shutil,platform
p=Path({R!r});old=Path({OLD!r});sys.path.insert(0,str(p))
from resource_gate import capacity
host=capacity();en=json.loads((p/'CASES.json').read_text())[0]['envelope'];assert host['effective_available_host_bytes']>=en['host_required'];assert shutil.disk_usage(p).free>=en['disk_required']
assert not (p/'square-4k-bfloat16').exists()
assert (p/'TileOPs/scripts/ascend_pair_check.py').is_file()
assert (p/'TileOPs/workloads/gemm.py').is_file()
for n in ['binding.py','adapter.py','resource_gate.py']:assert (p/n).read_bytes()==(old/n).read_bytes()
b=p/'private_provider/op_api/lib/libcust_opapi.so';raw=b.read_bytes();assert int.from_bytes(raw[18:20],'little')==183 and platform.machine()=='aarch64'
for f in p.glob('*.py'):ast.parse(f.read_text())
pins={{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for d in ['TileOPs','tools','private_provider'] for x in (p/d).rglob('*') if x.is_file()}}
for f in p.glob('*.py'):pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for n in ['run.sh','CASES.json']:pins[str(p/n)]=hashlib.sha256((p/n).read_bytes()).hexdigest()
(p/'FROZEN.json').write_text(json.dumps(dict(sha256=pins),indent=2))
receipt=dict(host=host,disk_free=shutil.disk_usage(p).free,abi=183,original_math_unchanged=True,output_absent=True,file_count=len(pins),provider_sha256=hashlib.sha256(raw).hexdigest())
(p/'CPU_PREFLIGHT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
r=py(pre);(P/'CPU_PREFLIGHT.json').write_text(json.dumps(r,indent=2));assert r['rc']==0,r
hb('ready_for_single_guarded_capture');print(r)
