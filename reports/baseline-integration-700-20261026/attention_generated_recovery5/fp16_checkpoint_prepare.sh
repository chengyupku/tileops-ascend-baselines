set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
npu-guard status --json > "$R/post_fp16_guard.json"
python3 - <<'PY'
import pathlib,json,os,hashlib,datetime
r=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5');b=r.parent/'attention_native_build5/attempt03/stage/source/build';gen=b/'binary/ascend950/gen'
st=json.loads((r/'post_fp16_guard.json').read_text());assert st['state']=='idle' and not st['reservation']['locked']
users=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  if p.name==str(os.getpid()):continue
  if str(b).encode() in (p/'cmdline').read_bytes() or str((p/'cwd').resolve()).startswith(str(b)) or any(str(f.resolve()).startswith(str(b)) for f in (p/'fd').iterdir()):users.append(p.name)
 except OSError:pass
assert not users,users
archive=r/'archived_fp16_generated';archive.mkdir();items=[]
for p in [*gen.glob('kernel_meta_*'),gen/'compile_stop.flag',gen/'compile_pids.txt']:
 if not p.exists():continue
 for f in (p.rglob('*') if p.is_dir() else [p]):
  if f.is_file():items.append({'path':str(f.relative_to(gen)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 p.rename(archive/p.name)
(r/'fp16_changed_condition.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'600s guard bound;624 proven completed original objects preserved for exact GNU make increment','users':users,'archived':items,'compiler_source_modified':False},indent=2))
s=(r/'RESUME_TARGET0.sh').read_text().replace('compile0.','compile0_checkpoint.')
(r/'RESUME_TARGET0_CHECKPOINT.sh').write_text(s);print('archived',len(items))
PY
