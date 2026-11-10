import json,pathlib
W=pathlib.Path('/opt/tileops-all-pairs/workspace'); P=W/'reports/baseline-integration-20261026/transformer_attention'; keys={tuple(r['key']) for r in json.loads((P/'SCOPE.json').read_text())}; base=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025'
paths=[base/'TARGET_SETS_CURRENT.json', W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/user_amnesty_0323/POLICY.json',W/'.cache/all-pairs/resume22_server2_transformer_attention01/FINAL_HANDOFF.json',base/'continuation0430/operational_recovery_0730']
def walk(d,path):
 if isinstance(d,dict):
  key=d.get('key') or [d.get('library',d.get('baseline')),d.get('op',d.get('operator')),d.get('case_id')]
  if isinstance(key,list) and len(key)==3 and all(isinstance(x,str) for x in key) and tuple(key) in keys:return [dict(location=path,row=d)]
  return sum((walk(v,path+'/'+k) for k,v in d.items()),[])
 if isinstance(d,list):
  if len(d)==3 and all(isinstance(x,str) for x in d) and tuple(d) in keys:return [dict(location=path,key=d)]
  return sum((walk(v,path+'/'+str(i)) for i,v in enumerate(d)),[])
 return []
out=[]
for p in paths:
 if p.is_file():
  out.extend(walk(json.loads(p.read_text()),str(p)))
 else:out.append({'path':str(p),'missing_or_directory':True})
(P/'EXACT_HISTORY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2)[:15000])
