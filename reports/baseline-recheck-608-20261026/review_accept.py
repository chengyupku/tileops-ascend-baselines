import json,pathlib,hashlib,datetime
R=pathlib.Path(__file__).resolve().parent

def accept(groups, note, overrides=None):
 rows=json.load(open(R/'SUBMITTED_DECISIONS.json'));p=R/'PARENT_ACCEPTED.json';old=json.load(open(p)) if p.exists() else [];seen={tuple(x['key']) for x in old};n=0
 for x in rows:
  if (x['key'][0],x['key'][1],x['decision']) not in groups or tuple(x['key']) in seen:continue
  v={'key':x['key'],'decision':x['decision'],'row_sha256':hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'note':note}
  if overrides and tuple(x['key'][:2]) in overrides:v['reason_override']=overrides[tuple(x['key'][:2])]
  old.append(v);seen.add(tuple(x['key']));n+=1
 p.write_text(json.dumps(old,ensure_ascii=False,indent=2));(R/'REVIEW.json').write_text(json.dumps({'accepted_count':len(old),'last_update':datetime.datetime.now(datetime.timezone.utc).isoformat(),'basis':'主agent已对照源码审阅的组及逐case合同，不是仅结构校验。'},ensure_ascii=False,indent=2));print('accepted_add',n,'total',len(old))

def accept_audit(directory,note,exclude_keys=None):
 src=R/directory/'AUDIT.json';d=json.load(open(src));rows=d.get('rows',d.get('results',[]));scope={tuple(x['key']) for x in json.load(open(R/directory/'SCOPE.json'))['rows']};p=R/'PARENT_ACCEPTED.json';old=json.load(open(p));by={tuple(x['key']):x for x in old};n=0
 for x in rows:
  k=tuple(x['key'])
  if exclude_keys and k in exclude_keys:continue
  assert k in scope and x['decision'] in ('要接','不要接') and x.get('reason') and x.get('evidence')
  v={'key':list(k),'decision':x['decision'],'source':str(src),'audited_row':x,'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'note':note}
  if k not in by:n+=1
  by[k]=v
 p.write_text(json.dumps(list(by.values()),ensure_ascii=False,indent=2));(R/'REVIEW.json').write_text(json.dumps({'accepted_count':len(by),'last_update':datetime.datetime.now(datetime.timezone.utc).isoformat(),'basis':'主审源代码核查及独立复核结果审阅；含专项补证填补原worker未决行。'},ensure_ascii=False,indent=2));print('audit_accepted_add',n,'total',len(by))
