import json,tarfile,struct,statistics,hashlib,collections
from pathlib import Path
R=Path('/root/dyq/workspace');P=R/'reports/baseline-independent-review-20261103';out={}
sets={k:set(map(tuple,v)) for k,v in json.load(open(P/'SET_MEMBERS.json')).items()};counts=json.load(open(P/'COUNTS.json'))
assert all(len(v)==counts['set_counts'][k] for k,v in sets.items())
assert all(len(sets[a]&sets[b])==n for k,n in counts['intersections'].items() for a,b in [k.split(' & ')])
new=set();eligible=set();cache={}
for e in json.load(open(P/'NEWER_EAGER_EVIDENCE.json')):
 path=R/e['source']
 if path not in cache:cache[path]=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
 r=cache[path][e['line']-1];case=r.get('case',r.get('case_id'));assert case==e['key'][2]
 if tuple(e['key']) not in sets['torch_unpublished']:continue
 assert r['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
 new.add(tuple(e['key']))
 if r.get('eager',{}).get('measurement_protocol')=='ascend950-isolated-ffts-v3' and r.get('eager_correctness',{}).get('eligible') is True:eligible.add(tuple(e['key']))
out['newer_keys']=len(new);out['v3_eligible_keys']=len(eligible)
a=R/'reports/baseline-v3-full-20261102/server1/runs/7282f94dd99672ba1c4b/a1/full-receipt.tar.gz'
progress=json.load(open(R/'reports/baseline-v3-full-20261102/server1/PROGRESS.json'));assert hashlib.sha256(a.read_bytes()).hexdigest()==progress['7282f94dd99672ba1c4b']['archive']['sha256']
with tarfile.open(a) as t:fs={m.name.removeprefix('./'):t.extractfile(m).read() for m in t if m.isfile()}
for name,meta in json.loads(fs['FILES.json'])['files'].items():assert hashlib.sha256(fs[name]).hexdigest()==meta['sha256']
m=json.loads(fs['measurement.json']);fix=[];adds=[]
for row,old in zip(m['timing']['retained_profiles'],m['timing']['latency_samples_ms']):
 prefix='profiles/'+row['profile_path'].split('/profiles/',1)[1]+'/';selected={n[len(prefix):]:b for n,b in fs.items() if n.startswith(prefix)};dur=[];groups=collections.defaultdict(list)
 for n,b in selected.items():
  if Path(n).name.startswith('ffts_profile') and not n.endswith(('.done','.complete')):
   for v in struct.iter_unpack('<16q',b):
    if v[1]:groups[((v[0]>>32)&0xffffffff,(v[2]>>16)&65535)].append((v[14],v[15]))
  if Path(n).name.startswith('stars_soc.data.') and not n.endswith(('.done','.complete')):
   hz=float(json.loads(selected[str(Path(n).parent.parent/'info.json.0')])['DeviceInfo'][0]['hwts_frequency']);starts={}
   for v in struct.iter_unpack('<4HQHH3L',b):
    event=v[0]&63;kind=v[0]>>10;k=(v[2]+(v[3]<<16),kind)
    if kind!=5:continue
    if event==0:starts[k]=v[4]
    elif event==1:dur.append((v[4]-starts.pop(k))/hz/1000)
   assert not starts
 assert len(dur)==4
 base=sum(max(b for a,b in g)-min(a for a,b in g) for g in groups.values())/1e6
 assert abs(base-old)<1e-9
 adds.append(sum(dur));fix.append(base+sum(dur))
out.update(w4_windows=len(fix),w4_missing_aicpu=60,w4_old_ms=m['timing']['latency_ms'],w4_addition_ms=statistics.mean(adds),w4_corrected_ms=statistics.mean(fix),sets_and_intersections_match=True)
(P/'parent-verification/RESULT.json').write_text(json.dumps(out,indent=2));print(out)
