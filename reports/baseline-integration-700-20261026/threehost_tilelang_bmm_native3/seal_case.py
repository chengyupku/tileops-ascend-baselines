from pathlib import Path
import subprocess,tarfile,json,hashlib,sys,os
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
tag='tilelang_BmmFwdOp_'+sys.argv[1]+(sys.argv[2] if len(sys.argv)>2 else '');remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tilelang_bmm_native3'
archive=P/(tag+'.tar.gz')
if not archive.exists():
 with archive.open('xb') as f:
  r=subprocess.run([str(ssh),'openlibing-DevEnv-661495','tar -czf - -C '+remote+' runs/'+tag],stdout=f,stderr=subprocess.PIPE);assert r.returncode==0,r.stderr
if not (P/'server1_evidence/runs'/tag).exists():
 with tarfile.open(archive) as t:t.extractall(P/'server1_evidence',filter='data')
out=P/'server1_evidence/runs'/tag;c=json.loads((out/'checker.json').read_text());assert c['status']=='verified' and c['inputs_preserved']
g=json.loads((P/(tag+'.guard.json')).read_text());assert g['returncode']==0
load=list(out.glob('native_load_*.json'));assert load
checks=[]
for f in load:
 r=json.loads(f.read_text());ptr=r['symbols']['__tvm_ffi_main'];assert ptr
 matches=[]
 for line in r['maps'].splitlines():
  cols=line.split();lo,hi=[int(x,16) for x in cols[0].split('-')]
  if lo<=ptr<hi:
   assert 'x' in cols[1];assert int(cols[4])==r['inode'];assert cols[-1]==r['path']
   matches.append(line)
 assert len(matches)==1
 artifact=out/'native_artifacts'/r['sha256'];assert hashlib.sha256(artifact.read_bytes()).hexdigest()==r['sha256']
 checks.append(dict(load=f.name,symbol='__tvm_ffi_main',pointer=ptr,map=matches[0],sha256=r['sha256'],binary_bytes=artifact.stat().st_size,stat_device=r['device'],maps_device=matches[0].split()[3]))
dispatch=list(out.glob('*.dispatch.json'));assert dispatch
for f in dispatch:
 r=json.loads(f.read_text());assert r['completed']
 # TVM unlinks the temporary exported.so after load; match the captured pointer
 # and inode in the still-live mapping, with its explicit '(deleted)' suffix.
 for native in checks:
  assert any(line.split()[:5]==native['map'].split()[:5] and line.removesuffix(' (deleted)')==native['map'] for line in r['maps_after'].splitlines())
 assert hashlib.sha256((out/(str(r['sequence'])+'.kernel.txt')).read_bytes()).hexdigest()==r['kernel_source_sha256']
record=dict(key=['tilelang','BmmFwdOp',sys.argv[1]],status='verified_native_complete',checker=c,guard=g,dispatch_count=len(dispatch),native_checks=checks,shared_registry_modified=False)
(P/(tag+'.SEALED.json')).write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['checker','guard']},indent=2))
