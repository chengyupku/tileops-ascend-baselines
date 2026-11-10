"""CPU-only final raw replay and immutable archive after successful guard exit."""
import pathlib,json,time,subprocess,sys,hashlib
P=pathlib.Path(__file__).resolve().parent;deadline=time.monotonic()+1200
while True:
 try:g=json.load(open(P/'native-group-v4-guard.json'))
 except (FileNotFoundError,json.JSONDecodeError):g={}
 if g.get('phase') in ('finished','released_without_launch'):
  assert g['workload_exit_code']==0 and g['performance_valid'],g
  # Controller copies the shared receipt to members after guard returns.
  members=json.load(open(P/'NATIVE_GROUP_V4_MEMBERS.json'))
  if len(members)==8 and all((pathlib.Path(r['out'])/'guard.json').exists() for r in members):break
 if time.monotonic()>deadline:raise RuntimeError('native final guard not ready')
 time.sleep(1)
subprocess.run([sys.executable,str(P/'audit_native.py')],check=True)
archive=pathlib.Path('/opt/aicpu-fix-native-evidence-20261108.tar.gz')
names=['native-v2','native-v3','native-v4','NATIVE_RESULTS.json','NATIVE_RESULTS.csv','NATIVE_GROUP_MEMBERS.json','NATIVE_GROUP_FAILURE.json','native-group-guard.json','NATIVE_GROUP_V3_MEMBERS.json','NATIVE_GROUP_V3_FAILURE.json','native-group-v3-guard.json','NATIVE_GROUP_V4_MEMBERS.json','native-group-v4-guard.json','native-group.log','native-group-v3.log','native-group-v4.log','native-v2-batch.log','native_group.py','native_group_v3.py','native_group_v4.py','start_native_group.py','start_native_group_v3.py','start_native_group_v4.py','audit_native.py','probe_constants.py','probe_constants_launch.sh','CONSTANT_PROBES.json','constant-probe-guard.json','constant-probe.log']
names += [p.name for p in P.glob('constant-probe-*') if p.is_dir()]
subprocess.run(['tar','--exclude=__pycache__','-czf',str(archive),'-C',str(P),*names],check=True)
h=hashlib.sha256(archive.read_bytes()).hexdigest();(P/'NATIVE_ARCHIVE.json').write_text(json.dumps({'archive':str(archive),'sha256':h,'raw_windows_verified':150,'pairs':10},indent=2));print(h,str(archive),flush=True)
