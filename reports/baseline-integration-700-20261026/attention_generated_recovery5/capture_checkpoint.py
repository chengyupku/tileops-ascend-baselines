import pathlib,json,hashlib,shlex,os,time,shutil,subprocess,datetime,sys
R=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5')
G=R.parent/'attention_native_build5/attempt03/stage/source/build/binary/ascend950/gen'
M=json.loads(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert M['owner']=='baseline700-20261026/attention_generated_recovery5' and M['phase']=='running'
C=R/'checkpoints'/sys.argv[1];C.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
active=set()
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  a=(p/'cmdline').read_bytes().decode().split('\0')
  if a and 'bisheng' in a[0] and '-o' in a:active.add(a[a.index('-o')+1])
 except (OSError,UnicodeError):pass
rows=[];inputs={};cutoff=time.time()-15
for p in list(G.glob('kernel_meta_*/kernel_meta/*.i'))+list(G.glob('kernel_meta_*/kernel_meta/*kernel.cpp'))+list(G.glob('kernel_meta_*/kernel_meta/*tiling_data.h')):
 inputs[str(p)]=sha(p)
for m in [*G.glob('kernel_meta_*/kernel_meta/*mk'),*R.glob('archived_serial_generated/kernel_meta_*/kernel_meta/*mk')]:
 lines=m.read_text().splitlines()
 for i,l in enumerate(lines):
  if not l.startswith('\t'):continue
  a=shlex.split(l)
  if '-o' not in a or 'bisheng' not in a[0]:continue
  p=pathlib.Path(a[a.index('-o')+1])
  if p.suffix!='.o' or not p.is_file() or str(p) in active or p.stat().st_mtime>=cutoff:continue
  if a[0] not in inputs:inputs[a[0]]=sha(a[0])
  check=subprocess.run(['/usr/bin/readelf','-h',str(p)],capture_output=True,text=True)
  assert check.returncode==0 and 'REL (Relocatable file)' in check.stdout,(p,check.stdout)
  h=sha(p);dest=C/p.name;shutil.copy2(p,dest);assert sha(dest)==h and sha(p)==h
  rows.append({'output':str(p),'saved':p.name,'sha256':h,'recipe_sha256':hashlib.sha256(l.encode()).hexdigest(),'target':lines[i-1][:-1],'mtime':p.stat().st_mtime})
manifest={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'guard_metadata':M,'inputs_sha256':inputs,'completed':rows,'excluded_live_outputs':sorted(active),'completion_evidence':'Valid stable ELF, older than15seconds, not output of any active bisheng; compiler make still running without reported recipe failure; original recipe retained exactly'}
(C/'manifest.json').write_text(json.dumps(manifest,indent=2));print('checkpoint',C,'completed',len(rows))
