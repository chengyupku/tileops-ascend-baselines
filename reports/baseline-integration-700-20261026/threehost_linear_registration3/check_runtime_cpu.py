"""CPU-only differential preflight; no native/Python device libraries are loaded."""
import pathlib,json,hashlib,subprocess,datetime,platform,sys,shutil,struct
O=pathlib.Path(__file__).resolve().parent;B=O.parent;S=B/'threehost_linear_registration8'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
new=read(O/'RUNTIME_PINS.json');old=read(S/'RUNTIME_PINS.json')['prefill-8k-bfloat16']['runtime'];receipt=read(S/'RUNTIME_CPU_CHECK.json')
assert receipt['passed'] and receipt['x86_64']['mismatches']==[]
union={}
for v in new.values():
 assert v['host']==dict(hostname=platform.node(),architecture=platform.machine())
 for p,h in v['runtime'].items():
  assert p not in union or union[p]==h
  union[p]=h
changed={p:h for p,h in union.items() if old.get(p)!=h}
for p,h in changed.items():assert sha(p)==h,p
pin=next(iter(new.values()));recipe=pin['recipe'];sdk=pathlib.Path(pin['environment']['ASCEND_HOME_PATH'])
paths=[recipe['opapi'],recipe['nnopbase'],pin['python_prefix']+'/bin/python',str(sdk/'tools/bisheng_compiler/bin/bisheng'),str(sdk/'opp'),str(B/'threehost_linear49/snapshot/TileOPs/scripts/ascend_pair_check.py')]
for p in paths:assert pathlib.Path(p).exists(),p
# Actual host exports/ELF/ABI checks read ELF bytes and start Python with site disabled.
for p in [recipe['opapi'],recipe['nnopbase']]:
 raw=pathlib.Path(p).read_bytes()[:20];assert raw[:4]==b'\x7fELF' and struct.unpack('<H',raw[18:20])[0]==62
 assert sha(p)==union[str(pathlib.Path(p).resolve())] if str(pathlib.Path(p).resolve()) in union else sha(p)==union[p]
exports=subprocess.run(['nm','-D',recipe['opapi']],text=True,capture_output=True,check=True).stdout
for sym in ('aclnnChunkGatedDeltaRule','aclnnChunkGatedDeltaRuleGetWorkspaceSize'):assert sym in exports
abi=subprocess.run([pin['python_prefix']+'/bin/python','-I','-S','-B','-c','import sys,sysconfig,json;print(json.dumps([sys.version_info[:3],sysconfig.get_config_var("SOABI")]))'],capture_output=True,text=True,check=True)
assert 'cpython-312-x86_64' in abi.stdout
# Exact wrapper source differs from sibling only as proved by the new execution receipts.
source=B/'threehost_linear_remaining4'
wrappers={}
for n in ('binding.py','direct_acl.py','entry.py'):
 p=source/'package_local'/n;wrappers[n]=sha(p)
 for run in ('delta_12k','delta_16k','delta_ragged_fixed'):
  assert read(source/run/'FROZEN.json')[str(p)]==wrappers[n]
# Recorded original cache route worked. Deployment uses /proc/self/cwd/<cache> under fixed private cwd.
assert pathlib.Path('/proc/self/cwd').resolve()==pathlib.Path.cwd()
result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=pin['host'],runtime_entries=len(union),reused_unchanged_entries=len(union)-len(changed),delta_hashed=changed,reused_receipt=str(S/'RUNTIME_CPU_CHECK.json'),reused_receipt_sha256=sha(S/'RUNTIME_CPU_CHECK.json'),reuse_limit='Unchanged identity reused from earlier CPU hash receipt; no claim of freshly hashing all SDK files',wrapper_hashes=wrappers,paths_checked=paths,python_abi=json.loads(abi.stdout),native_exports=['aclnnChunkGatedDeltaRule','aclnnChunkGatedDeltaRuleGetWorkspaceSize'],short_cache_route='/proc/self/cwd/<cache>; launch fixes cwd to exclusive run directory',three_host_capability='No device submission; exact prior successful host aa/x86_64 only. Other hosts not admitted or probed.',device_imports=0,npu_attempts=0)
(O/'RUNTIME_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,runtime_entries=len(union),reused=len(union)-len(changed),delta_hashed=len(changed))))
