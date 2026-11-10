"""Offline admission of explicitly authorized six-output device reference only.
Shared collector is untouched; only already cleaned server2 receipts are updated.
"""
import json,pathlib,tarfile,tempfile,statistics,time
from collector import P,REMOTE,sha,parser,parser32,put

def main():
 proof=P/'SOURCES.v14-server2.json'
 sources=json.loads(proof.read_text())
 for name in ['worker_server2_v14.py','adapters/delta_chunk_server2.py','launch_server2_v14.sh']:
  assert sha(P/name)==sources[name],name
 for report in (P/'server2/runs').glob('*/a*/ARCHIVE.json'):
  receipt=json.loads(report.read_text())
  if receipt.get('accepted') or not receipt.get('cleaned'):continue
  dest=report.parent;tid=dest.parent.name;attempt=dest.name
  if not (dest/'measurement.json').exists():continue
  m=json.loads((dest/'measurement.json').read_text())
  if m['key'][1]!='DeltaNetChunkFwdOp' or m['source_sha256']!=sha(proof):continue
  archive=pathlib.Path(receipt['archive'])
  assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
  with tempfile.TemporaryDirectory(prefix='server2-v14-replay-') as tmp:
   root=pathlib.Path(tmp)
   with tarfile.open(archive) as tar:
    for member in tar.getmembers():
     p=pathlib.PurePosixPath(member.name);assert not p.is_absolute() and '..' not in p.parts and (member.isfile() or member.isdir())
    tar.extractall(root,filter='data')
   for name,entry in json.loads((root/'FILES.json').read_text())['files'].items():
    assert (root/name).stat().st_size==entry['size'] and sha(root/name)==entry['sha256'],name
   m=json.loads((root/'measurement.json').read_text());g=json.loads((root/'guard.json').read_text());c=json.loads((root/'correctness.json').read_text());task=json.loads((root/'task.json').read_text());a=json.loads((root/'interface_adaptation.json').read_text())
   assert m['source_sha256']==sha(proof) and a['all_six_outputs'] and a['cpu_oracle_unchanged']
   assert m['status']=='measured_pending_guard' and m['key']==task['key'] and m['task_id']==tid
   assert g['performance_valid'] and g['phase']=='finished' and g['workload_exit_code']==g['exit_code']==0 and g['owner']=='torch1135/server2/'+tid
   assert c['status']=='verified' and c['inputs_preserved'] and m['inputs_preserved'] and sha(root/'correctness.json')==m['correctness_sha256']
   assert c['check']['checked_outputs']==c['check']['total_outputs']==6
   assert (m['hardware']['cube'],m['hardware']['vector'])==(32,64)
   samples=m['timing']['latency_samples_ms'];profiles=m['timing']['retained_profiles'];assert len(samples)==len(profiles)==15
   values=[]
   for row,sample in zip(profiles,samples):
    relative=pathlib.Path(row['profile_path']).relative_to(pathlib.Path(REMOTE)/'server2/runs'/tid/attempt);directory=root/relative
    assert parser.trace_digest(directory)==row['profile_trace_sha256']
    decoder=parser32 if row['parser_revision']==parser32.PARSER_REVISION else parser
    parsed=decoder.parse_profile(directory,1);assert parsed['parser_revision']==row['parser_revision'] and abs(parsed['latency_ms']-sample)<1e-9;values.append(parsed['latency_ms'])
   assert abs(statistics.mean(values)-m['timing']['latency_ms'])<1e-9
  backup=dest/'ARCHIVE.before-v14-admission.json'
  if not backup.exists():backup.write_bytes(report.read_bytes())
  receipt.update(accepted=True,error=None,measurement={'latency_ms':statistics.mean(values),'key':m['key'],'raw_windows':15},at=time.time(),admission={'validator':str(pathlib.Path(__file__).resolve()),'validator_sha256':sha(pathlib.Path(__file__)),'source_sha256':sha(proof),'basis':'User authorized new six-output torch NPU reference; original independent checker and full raw replay pass'})
  put(report,receipt)
  print(tid,attempt,receipt['measurement'],flush=True)
if __name__=='__main__':main()
