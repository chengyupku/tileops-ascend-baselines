from pathlib import Path
import json,hashlib,re,datetime
W=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gdn_fp16_capture14')
results=[]
for a in [W/'attempt01_decode',W/'attempt02_varlen']:
 f=json.loads((a/'FROZEN.json').read_text());check=[]
 for path,r in f['files'].items():
  p=Path(path);check.append(dict(path=path,unchanged=hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']))
 raw=(a/'raw.log').read_text();paths=set(re.findall(r'bin path: (\S+ChunkKdaFwd_\S+\.o)',raw))
 binaries=[]
 for path in paths:
  p=Path(path);h=hashlib.sha256(p.read_bytes()).hexdigest();binaries.append(dict(path=path,resolved=str(p.resolve()),sha256=h,hash_was_prefrozen=any(x['sha256']==h for x in f['files'].values())))
 maps=(a/'runtime.maps').read_text();mapped={line.split()[-1] for line in maps.splitlines() if '/' in line and '.so' in line}
 relevant=[p for p in mapped if any(x in p for x in ['libopapi_transformer','libnnopbase','libtiling_device_transformer','libascendcl','libruntime','libtorch_npu'])]
 libs=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in sorted(relevant) if Path(p).is_file()]
 r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt=a.name,frozen_files_unchanged=all(x['unchanged'] for x in check),checks=check,actual_dispatched_binaries=binaries,mapped_runtime_post_hashes=libs,post_hash_not_pre_attestation=True)
 (a/'POST_AUDIT.json').write_text(json.dumps(r,indent=2));results.append(r)
print(json.dumps(results,indent=2))
