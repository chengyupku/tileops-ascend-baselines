import json,hashlib,fcntl,sys,types,builtins,importlib.util,difflib,datetime,time,os
from pathlib import Path
O=Path(__file__).resolve().parent;remote='--remote' in sys.argv
repo=Path('/opt/tileops-all-pairs/workspace/TileOPs' if remote else '/root/dyq/workspace/TileOPs');D=repo/'benchmarks/ascend_all_pairs/ops_transformer';name='integration700_gdn_chunk_registration16';host='server2' if remote else 'local'
def sha(b):return hashlib.sha256(b).hexdigest()
def active():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:a=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any(Path(x.decode(errors='replace')).name=='ascend_pair_check.py' for x in a):found.append(p.name)
 if found:raise RuntimeError('Fullchecker active; no publication: '+str(found))
original=builtins.__import__
def deny(n,*a,**kw):
 if n.split('.')[0] in ('torch','torch_npu','acl'):raise RuntimeError('CPU registry probe blocks native imports '+n)
 return original(n,*a,**kw)
builtins.__import__=deny
sys.dont_write_bytecode=True
# Actual module files, under an isolated package name; no executable test switch.
def load_registry(tag,raw):
 mod=types.ModuleType(tag);mod.__file__=str(D/'__init__.py');mod.__path__=[str(D)];mod.__package__=tag;sys.modules[tag]=mod
 exec(compile(raw,mod.__file__,'exec'),mod.__dict__);return mod
start=time.monotonic()
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);active()
 p=D/'__init__.py';before=p.read_bytes();prior=load_registry('cpu_before',before)
 module=(O/(name+'.py')).read_bytes();ns={};exec(compile(module,str(O/(name+'.py')),'exec'),ns)
 if remote:
  checks={}
  for path,pin in ns['RUNTIME_PINS'].items():
   actual=sha(Path(path).read_bytes());assert actual==pin,path;checks[path]=actual
  ns['_snapshot']()
  (O/'REMOTE_PIN_CHECK.json').write_text(json.dumps({'checked':len(checks),'all_match':True,'pins':checks},indent=2)+'\n')
 addition=f'\n# Frozen original whole ChunkGatedDeltaRule exact16; retain all prior routes.\nfrom .{name} import wrap as _wrap_gdn_chunk16\nPROVIDERS["GDNFwdOp"] = _wrap_gdn_chunk16(PROVIDERS.get("GDNFwdOp"))\n'.encode()
 assert name.encode() not in before,'already registered; inspect existing receipt'
 (O/(host+'_provider.before.py')).write_bytes(before)
 (D/(name+'.py')).write_bytes(module);p.write_bytes(before+addition)
 after=p.read_bytes();assert after==before+addition
 post=load_registry('cpu_after',after);frozen=sys.modules['cpu_after.'+name]
 assert (D/(name+'.py')).read_bytes()==module
 # Intercept only build in the test namespace to observe exact routing; original build is proven separately.
 frozen.build=lambda c:('exact16',c.id)
 for cid in frozen.CASE_PARAMS:assert post.PROVIDERS['GDNFwdOp'](types.SimpleNamespace(id=cid))==('exact16',cid)
 prev=post.PROVIDERS['GDNFwdOp'].__closure__
 captured=[c.cell_contents for c in prev if callable(c.cell_contents)]
 assert len(captured)==1
 previous=captured[0]
 # The wrapper retains the actual previous function by identity.
 assert previous.__module__.startswith('cpu_after')
 fallback_calls=[]
 assert frozen.wrap(lambda c:fallback_calls.append(c.id) or 'old')(types.SimpleNamespace(id='unmatched'))=='old';assert fallback_calls==['unmatched']
 assert set(prior.PROVIDERS)==set(post.PROVIDERS)
 for op,fn in prior.PROVIDERS.items():
  if op=='GDNFwdOp':continue
  got=post.PROVIDERS[op];assert fn.__name__==got.__name__ and fn.__module__.replace('cpu_before','cpu_after')==got.__module__,op
 (O/(host+'_registration.diff')).write_text(''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile=str(p)+':before',tofile=str(p)+':after')))
 (O/(host+'_REGISTRATION.json')).write_text(json.dumps({'registered':16,'provider_before_sha256':sha(before),'provider_after_sha256':sha(after),'module_sha256':sha(module),'provider_routes_preserved':len(prior.PROVIDERS),'exact_routes_checked':16,'fallback_preserved':True,'lock':'/tmp/baseline700-provider-ops_transformer.lock','cpu_only':True,'seconds':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
print(host,'registered16 and CPU disk routes verified')
