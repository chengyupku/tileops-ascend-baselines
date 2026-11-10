from pathlib import Path
import subprocess,sys,json,shlex
P=Path(__file__).resolve().parent;SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';h=sys.argv[1];rp='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker02'
d=P/h;d.mkdir(exist_ok=True)
cmd=[SSH,h,'tar -C '+shlex.quote(rp)+" --exclude='*.pt' --exclude='*.so' --exclude='TMPDIR' --exclude='ASCEND_CACHE_PATH' --exclude='XDG_CACHE_HOME' -czf - runs"]
a=subprocess.Popen(cmd,stdout=subprocess.PIPE);b=subprocess.run(['tar','-C',str(d),'-xzf','-'],stdin=a.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE);a.stdout.close();rc=a.wait();print(json.dumps(dict(host=h,fetch_rc=rc,extract_rc=b.returncode,stderr=b.stderr.decode()[-500:])))
# Preserve byte-exact tensor archives locally, then reclaim only our verified copies
# from remote private storage; server1 has limited free space.
import hashlib
probe="from pathlib import Path;import json; p=Path("+repr(rp)+");print(json.dumps([str(f.relative_to(p)) for f in p.glob('runs/*/inputs.pt')]+[str(f.relative_to(p)) for f in p.glob('runs/*/launch_*/*.so')]))"
r=subprocess.run([SSH,h,'python3 -c '+shlex.quote(probe)],capture_output=True,text=True,check=True)
files=json.loads(r.stdout);receipt=[]
seen={json.loads(line)['remote'] for line in (P/'INPUT_ARTIFACT_RELOCATIONS.jsonl').read_text().splitlines()} if (P/'INPUT_ARTIFACT_RELOCATIONS.jsonl').exists() else set()
for rel in files:
 if rp+'/'+rel in seen:continue
 local=d/rel
 if local.exists():continue
 local.parent.mkdir(parents=True,exist_ok=True)
 remote=rp+'/'+rel
 hcode="import hashlib;from pathlib import Path;p=Path("+repr(remote)+");f=p.open('rb');h=hashlib.sha256();[h.update(b) for b in iter(lambda:f.read(1048576),b'')];print(h.hexdigest())"
 expected=subprocess.run([SSH,h,'python3 -c '+shlex.quote(hcode)],capture_output=True,text=True,check=True).stdout.strip()
 if rel.endswith('/inputs.pt'):
  size=int(subprocess.run([SSH,h,'stat -c %s '+shlex.quote(remote)],capture_output=True,text=True,check=True).stdout.strip())
  receipt.append(dict(remote=remote,local=None,sha256=expected,size=size,remote_input_relocated=False,retained_on_execution_host=True))
  continue
 with local.open('xb') as stream:
  fetched=subprocess.run([SSH,h,'cat '+shlex.quote(remote)],stdout=stream)
 actual=hashlib.file_digest(local.open('rb'),'sha256').hexdigest()
 assert fetched.returncode==0 and actual==expected,(remote,actual,expected)
 removed=False
 if rel.endswith('/inputs.pt'):
  rmcode="import hashlib;from pathlib import Path;p=Path("+repr(remote)+");f=p.open('rb');h=hashlib.sha256();[h.update(b) for b in iter(lambda:f.read(1048576),b'')];f.close();assert h.hexdigest()=="+repr(actual)+";p.unlink()"
  subprocess.run([SSH,h,'python3 -c '+shlex.quote(rmcode)],check=True);removed=True
 receipt.append(dict(remote=remote,local=str(local),sha256=actual,size=local.stat().st_size,remote_input_relocated=removed))
if receipt:
 journal=P/'INPUT_ARTIFACT_RELOCATIONS.jsonl'
 with journal.open('a') as f:
  for r in receipt:f.write(json.dumps(r)+'\n')
 print(json.dumps(dict(host=h,artifacts_preserved=len(receipt))),flush=True)
