"""Server1-only offline acceptance for explicitly authorized new reference.
Runs after shared collector closes its archive; never touches the remote or NPU.
All original collector checks are retained except the obsolete CPU-route ban,
replaced by exact authorized source/provenance and six-output requirements.
"""
import pathlib,json,time,sys,tarfile,tempfile,textwrap,statistics,hashlib,fcntl
P=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import collector as original
COLLECTOR_SHA='8472ef0a96f4e9ec212334d24e22327d2b8b34b531f1313d2f2b7c13e65ffd5a'

def replay():
 lock=(P/'server1/delta-replay.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source=(P/'collector.py').read_text()
 assert hashlib.sha256(source.encode()).hexdigest()==COLLECTOR_SHA,'collector changed: re-review required'
 checks=textwrap.dedent(source[source.index('    m=json.loads'):source.index('   except Exception as e:error=')])
 ban="assert m['key'][1]!='DeltaNetChunkFwdOp', 'Known CPU-only reference route: requires all-output NPU counterpart, see CPU_FALLBACK_ROUTE_AUDIT.json'"
 assert checks.count(ban)==1;checks=checks.replace(ban,'assert m["key"][1]=="DeltaNetChunkFwdOp"')
 allowed={r['id'] for r in json.loads((P/'server1/PLAN.json').read_text())['rows'] if r['key'][1]=='DeltaNetChunkFwdOp'}
 manifest=P/'SOURCES.v10.server1.json';manifest_hash=original.sha(manifest)
 for tid in allowed:
  for f in (P/'server1/runs'/tid).glob('*/ARCHIVE.json'):
   receipt=json.loads(f.read_text())
   if receipt.get('accepted') or not receipt.get('cleaned') or 'Known CPU-only reference route' not in (receipt.get('error') or ''):continue
   attempt=f.parent.name
   try:
    with tempfile.TemporaryDirectory() as tmp:
     root=pathlib.Path(tmp);archive=pathlib.Path(receipt['archive']);assert original.sha(archive)==receipt['sha256']
     with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
     for n,d in json.loads((root/'FILES.json').read_text())['files'].items():assert (root/n).stat().st_size==d['size'] and original.sha(root/n)==d['sha256']
     meta=json.loads((root/'new_reference.json').read_text());assert meta['implementation']=='adapters/delta_chunk_six_server1_v1.py';assert meta['sha256']==original.sha(P/meta['implementation']);assert meta['outputs']==['o','S','Aw','Au','w','u'];assert all(meta[k] for k in ('all_computation_on_input_device','full_call_timed','checker_unchanged'));assert not meta['historical_reference']
     m=json.loads((root/'measurement.json').read_text());assert m['source_sha256']==manifest_hash
     c=json.loads((root/'correctness.json').read_text());assert c['check']['checked_outputs']==6 and c['check']['total_outputs']==6
     ns=dict(root=root,host='server1',tid=tid,attempt=attempt,json=json,pathlib=pathlib,statistics=statistics,P=P,REMOTE=original.REMOTE,sha=original.sha,parser=original.parser,parser32=original.parser32)
     exec(checks,ns)
     backup=f.with_name('ARCHIVE.before_authorized_delta.json');assert not backup.exists();backup.write_bytes(f.read_bytes())
     receipt.update(accepted=True,error=None,measurement=ns['measurement'],at=time.time(),reference_provenance=meta,offline_replay_reason='Explicit user-approved new six-output device reference; all original collector identity/guard/checker/raw checks passed')
     original.put(f,receipt);print('accepted',tid,attempt,flush=True)
   except Exception as exc:
    original.put(f.parent/'AUTHORIZED_REPLAY_ERROR.json',{'at':time.time(),'error':repr(exc)});raise

if __name__=='__main__':replay()
