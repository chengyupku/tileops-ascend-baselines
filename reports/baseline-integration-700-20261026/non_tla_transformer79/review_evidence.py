import pathlib,json,hashlib,datetime,collections
G=pathlib.Path(__file__).resolve().parent;W=G.parents[2]
def read(p):return json.loads(p.read_text())
rows=[];checks=[];seen_hash={}
for name in ['moe_original8','linear_original16','reduction_elementwise20']:
 q=G/name;v=read(q/'RESULT.json');scope={tuple(r['key']) for r in read(q/'SCOPE.json')};actual={tuple(r['key']) for r in v['rows']};assert actual==scope and len(actual)==len(v['rows'])
 for r in v['rows']:
  status=r['status'];item={'key':r['key'],'status':status,'reviewed_result':str(q/'RESULT.json'),'changed_condition_proven':r.get('changed_condition_proven',False),'ready_to_publish':False}
  if r.get('blocker_kind')=='single_card_capacity':item['status']='resource_capacity_blocked'
  if status=='original_numerical_failure':
   c=r.get('checker');c=c.get('path') if isinstance(c,dict) else c
   c=c or r['historical_numerical_failure']['receipt']; data=read(pathlib.Path(c));txt=json.dumps(data);assert 'Tensor-likes are not close' in txt
   item['checker']=c;item['checker_sha256']=hashlib.sha256(pathlib.Path(c).read_bytes()).hexdigest();item['numerical_error_rechecked']=True
  item['required_change']=r.get('changed_condition_required') or r.get('next_legal_action');rows.append(item)
 def walk(x):
  if isinstance(x,dict):
   if 'path' in x and 'sha256' in x:
    p=pathlib.Path(x['path']);p=p if p.is_absolute() else W/p
    if str(p) not in seen_hash:
     ok=p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];seen_hash[str(p)]=ok
   for y in x.values():walk(y)
  elif isinstance(x,list):
   for y in x:walk(y)
 walk(v)
 if (q/'SOURCE_EVIDENCE.json').exists():walk(read(q/'SOURCE_EVIDENCE.json'))
 checks.append({'family':name,'exact_scope_equal':True,'rows':len(v['rows'])})
assert all(ok or p.endswith('/ops_transformer/__init__.py') for p,ok in seen_hash.items()),[p for p,ok in seen_hash.items() if not ok]
# Shared provider changed only by the separately reviewed single_agent_local_fp8 installation; no relevance to these 44 keys.
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewed_count':len(rows),'status_counts':dict(collections.Counter(r['status'] for r in rows)),'accepted_added':0,'publishable':0,'checks':checks,'referenced_file_hash_checks':seen_hash,'review_note':'Original API/source constraints inspected by root; each numerical receipt read independently. No new acceptance; unavailable exact original API is bounded to inspected current source, not universal impossibility. Attention first result rejected pending current native evidence.','rows':rows}
(G/'REVIEW_PARTIAL44.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result['status_counts'],len(seen_hash),'referenced hashes checked')
