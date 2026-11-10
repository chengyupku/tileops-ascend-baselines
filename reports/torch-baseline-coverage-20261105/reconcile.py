import json,pathlib,collections
P=pathlib.Path(__file__).resolve().parent;R=P.parent
rows=json.loads((P/'CURRENT_CASES.json').read_text())['rows'];known={}
for r in json.loads((R/'normalization-pr-20261104/frozen-baseline.json').read_text())['rows']:
 if r['library']=='torch-eager':known[(r['op'],r['case'])]={'case':r['workload'],'origin':'published1135+527','protocol':r['protocol']}
for r in json.loads((R/'torch-baseline400-20261104/server1/PLAN.json').read_text())['rows']:
 d=json.loads((R/'torch-baseline400-20261104/server1/PROGRESS.json').read_text()).get(r['id'],{})
 if d.get('accepted'):known[tuple(r['key'][1:])]={'case':r['case'],'origin':'torch400','archive':d['archive']}
def stable(x):
 if isinstance(x,dict):return {k:stable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [stable(v) for v in x]
 if isinstance(x,float) and x!=x:return 'nan'
 if x in ('inf','Infinity','float inf') or isinstance(x,float) and x==float('inf'):return 'inf'
 if x in ('-inf','-Infinity') or isinstance(x,float) and x==-float('inf'):return '-inf'
 return x
result=[]
changedops={'GemmW4A8FwdOp','FusedAddRMSNormPerTokenQuantFwdOp','FusedQKNormRoPEFwdOp'}
for r in rows:
 k=tuple(r['key'][1:]);old=known.get(k)
 if not old:status='missing_current_success'
 elif stable(old['case']['params'])!=stable(r['case']['params']):status='changed_parameters'
 elif k[0] in changedops:status='changed_reference_semantics'
 else:status='matching_existing_candidate'
 result.append(dict(r,status=status,existing=old))
(P/'COVERAGE_AUDIT.json').write_text(json.dumps({'operators':len(set(r['key'][1] for r in rows)),'cases':len(rows),'counts':dict(collections.Counter(r['status'] for r in result)),'rows':result,'scope_note':'Both old/current manifests enumerate205 operators; user206 total remains inventory discrepancy, no case silently omitted. Matching_existing_candidate requires dependency review, not blanket new验收.'},indent=2))
for status in sorted(set(r['status'] for r in result)):
 print(status,dict(collections.Counter(r['key'][1] for r in result if r['status']==status)))
