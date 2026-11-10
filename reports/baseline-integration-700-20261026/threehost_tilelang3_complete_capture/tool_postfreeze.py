import pathlib,json,hashlib,shutil,datetime
P=pathlib.Path(__file__).resolve().parent;records=[]
for f in (P/'runs').glob('*/compiler_subprocess.jsonl'):
 for line in f.read_text().splitlines():
  r=json.loads(line)
  if not r.get('observed_executable'):continue
  p=pathlib.Path(r['executable']);data=p.read_bytes();h=hashlib.sha256(data).hexdigest();assert h==r['observed_executable']['sha256']
  dst=P/'frozen_content'/h
  if not dst.exists():dst.write_bytes(data);dst.chmod(0o444)
  st=p.stat();assert st.st_ino==r['observed_executable']['inode'] and st.st_dev==r['observed_executable']['device']
  records.append(dict(receipt=str(f.relative_to(P)),observed_utc=r['utc'],path=str(p),sha256=h,inode=st.st_ino,device=st.st_dev,post_freeze_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pre_cpu_frozen=False))
(P/'TOOL_POSTFREEZE.json').write_text(json.dumps(records,indent=2));print('post-frozen execution-hashed tool records',len(records))
