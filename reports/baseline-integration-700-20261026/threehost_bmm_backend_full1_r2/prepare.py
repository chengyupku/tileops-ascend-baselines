from control import *
import ast
old=P.parent/'threehost_bmm_input_backend1'
OLD=R.replace(P.name,old.name)
for n in ['adapter.py','binding.py','native_observation.py','resource_gate.py','CASES.json']:(P/n).write_bytes((old/n).read_bytes())
s=(old/'capture.py').read_text().replace(OLD,R)
s=s.replace("original_cpu_reference=c.cpu_reference",'''backend_events=[]
original_bmm=torch.bmm
original_backend=torch.backends.mkldnn.enabled
def backend_event(phase,reference=None):
 event=dict(phase=phase,time=time.time(),mkldnn_enabled=torch.backends.mkldnn.enabled,num_threads=torch.get_num_threads(),bmm_repr=repr(torch.bmm),bmm_identity=id(torch.bmm),bmm_unchanged=torch.bmm is original_bmm,native_events=list(native_events) if 'native_events' in globals() else [])
 if reference is not None:
  import inspect
  event.update(reference_repr=repr(reference),reference_source=inspect.getsource(reference),reference_file=inspect.getsourcefile(reference))
 backend_events.append(event);dump('BACKEND_VARIANT.json',dict(setting='torch.backends.mkldnn.enabled=False',original=original_backend,events=backend_events))
 print('BACKEND_STATE',json.dumps(event,default=str),flush=True)
backend_event('before_configuration')
torch.backends.mkldnn.enabled=False
torch.set_num_threads(4)
backend_event('configured')
original_cpu_reference=c.cpu_reference''')
s=s.replace("  output=reference(*args)","  backend_event('reference_before',reference)\n  assert torch.bmm is original_bmm and not torch.backends.mkldnn.enabled\n  output=reference(*args)\n  backend_event('reference_after',reference)")
s=s.replace("rc=c.main()", "try:\n rc=c.main()\nfinally:\n backend_event('before_restore')\n torch.backends.mkldnn.enabled=original_backend\n backend_event('restored')")
(P/'capture.py').write_text(s)
(P/'run.sh').write_text((old/'run.sh').read_text().replace(OLD,R))
for f in P.glob('*.py'):ast.parse(f.read_text())
r=py(f'''from pathlib import Path
import shutil
p=Path({R!r});old=Path({OLD!r});p.mkdir(exist_ok=False)
for n in ['TileOPs','tools','private_provider']:shutil.copytree(old/n,p/n,ignore=shutil.ignore_patterns('__pycache__'))
print('original closure copied')''');assert r['rc']==0,r
(P/'COPY_RECEIPT.json').write_text(json.dumps(r,indent=2))
for n in ['adapter.py','binding.py','native_observation.py','resource_gate.py','CASES.json','capture.py','run.sh']:upload(n)
pre=f'''from pathlib import Path
import sys,json,hashlib,ast,shutil,platform,subprocess
p=Path({R!r});old=Path({OLD!r});sys.path.insert(0,str(p))
from resource_gate import capacity
host=capacity();en=json.loads((p/'CASES.json').read_text())[0]['envelope'];assert host['effective_available_host_bytes']>=en['host_required'];assert shutil.disk_usage(p).free>=en['disk_required']
assert not (p/'square-4k-bfloat16').exists()
for n in ['binding.py','adapter.py','resource_gate.py']:assert (p/n).read_bytes()==(old/n).read_bytes()
b=p/'private_provider/op_api/lib/libcust_opapi.so';raw=b.read_bytes();assert int.from_bytes(raw[18:20],'little')==183 and platform.machine()=='aarch64'
pins={{}}
for d in ['TileOPs','tools','private_provider']:
 for x in (p/d).rglob('*'):
  if x.is_file():
   assert x.read_bytes()==(old/x.relative_to(p)).read_bytes(),str(x)
   pins[str(x)]=hashlib.sha256(x.read_bytes()).hexdigest()
for f in p.glob('*.py'):ast.parse(f.read_text());pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for n in ['run.sh','CASES.json']:pins[str(p/n)]=hashlib.sha256((p/n).read_bytes()).hexdigest()
(p/'FROZEN.json').write_text(json.dumps(dict(sha256=pins),indent=2))
receipt=dict(host=host,disk_free=shutil.disk_usage(p).free,abi=183,original_closure_byte_identical=True,output_absent=True,file_count=len(pins),provider_sha256=hashlib.sha256(raw).hexdigest(),runtime_launcher=Path('/opt/tileops-all-pairs/control/private_python.py').read_text())
(p/'CPU_PREFLIGHT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
r=py(pre);(P/'CPU_PREFLIGHT.json').write_text(json.dumps(r,indent=2));assert r['rc']==0,r
hb('ready_for_single_guarded_fullcase');print('CPU preflight passed')
