set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
npu-guard status --json > "$R/post_serial_guard.json"
python3 - <<'PY'
import pathlib,json,os,hashlib,datetime
r=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5'); b=r.parent/'attention_native_build5/attempt03/stage/source/build'; gen=b/'binary/ascend950/gen'
st=json.loads((r/'post_serial_guard.json').read_text());assert st['state']=='idle' and not st['reservation']['locked']
users=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes(); cwd=str((p/'cwd').resolve());fds=[str(f.resolve()) for f in (p/'fd').iterdir()]
  if p.name!=str(os.getpid()) and (str(b).encode() in cmd or cwd.startswith(str(b)) or any(x.startswith(str(b)) for x in fds)):users.append(p.name)
 except (OSError,PermissionError):pass
assert not users, users
archive=r/'archived_serial_generated';archive.mkdir()
items=[]
for p in [*gen.glob('kernel_meta_*'),gen/'compile_stop.flag',gen/'compile_pids.txt']:
 if not p.exists():continue
 files=list(p.rglob('*')) if p.is_dir() else [p];pins=[{'path':str(f.relative_to(gen)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'size':f.stat().st_size} for f in files if f.is_file()]
 p.rename(archive/p.name);items.extend(pins)
(r/'parallel_changed_condition.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'560 aiv tiling targets; serial throughput insufficient; supported make parallelism raised to 8 via new private wrapper','users':users,'archived_files':items,'old_source_changed':False},indent=2))
(r/'bin').mkdir(exist_ok=True)
s='''#!/opt/tileops-ci/runtime/bin/python3
import os,sys
args=[]; skip=False
for a in sys.argv[1:]:
 if skip: skip=False; continue
 if a in ('-j','--jobs'): skip=True; continue
 if a.startswith('-j') or a.startswith('--jobs='): continue
 args.append(a)
os.execv('/usr/bin/make',['/usr/bin/make',*args,'-j8'])
'''
for n in ['make','gmake']:
 p=r/'bin'/n;p.write_text(s);p.chmod(0o755)
s=(r/'RESUME_TARGET1.sh').read_text().replace('compile1.','compile1_parallel.').replace('export PATH="$B/stage/bin:$PATH"','export PATH="$R/bin:$B/stage/bin:$PATH"').replace('"$B/stage/bin/make"','"$R/bin/make"')
(r/'RESUME_TARGET1_PARALLEL.sh').write_text(s)
print('archived files',len(items),'wrapper_sha256',hashlib.sha256((r/'bin/make').read_bytes()).hexdigest())
PY
