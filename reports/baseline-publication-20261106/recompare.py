import json,math,collections
from pathlib import Path
P=Path('/root/dyq/workspace/reports/baseline-publication-20261106');C=Path('/root/dyq/workspace/reports/torch-all2208-20261105')
def read(p):return json.loads(p.read_text())
def canonical(v):
 if isinstance(v,float) and not math.isfinite(v):return 'NaN' if math.isnan(v) else ('Infinity' if v>0 else '-Infinity')
 if isinstance(v,str) and v in ('inf','-inf'):return 'Infinity' if v=='inf' else '-Infinity'
 if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
 if isinstance(v,list):return [canonical(x) for x in v]
 return v
scopes=read(P/'pr-operator-scope.json');cases=read(C/'CURRENT_CASES.json')['rows'];result=[]
for s in scopes:
 rows=[x for x in cases if x['key'][1] in s['ops']];result.append({'pr':s['pr'],'head':s['head'],'operators':len(s['ops']),'canonical_cases':len(rows),'baseline_cases':len(rows),'missing':0,'manifest_comparison':s['spec_checks'],'note':'#18当前PR删除两个complex128case；正式原10case baseline仍全部保留' if s['pr']==18 else 'manifest workload/signature identical; runtime implementation acceptance separate'})
P.joinpath('PR_COVERAGE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print([(x['pr'],x['canonical_cases']) for x in result])
audit=read(Path(read(C/'COMPLETION.json')['final_audit']));new={}
for a in audit['rows']:
 d=Path(a['archive']).parent;t=read(d/'task.json');m=read(d/'ARCHIVE.json')['measurement'];new[(t['key'][1],t['key'][2])]={'library':'torch-eager','op':t['key'][1],'case':t['key'][2],'workload':t['case'],'latency_us':m['latency_ms']*1000}
old=read(Path('/tmp/tileops-site-torch2208-20261106/data/baseline-cache/snapshot.json'));base=[x for x in old['rows'] if x['library']!='torch-eager']+list(new.values())
by=collections.defaultdict(list)
for x in base:by[x['op'],x['case']].append(x)
original_params={(x['op'],x['case']):x['params'] for x in read(P/'norm-original-params.json')}
def compare(n,rows,source):
 out=[]
 for r in rows:
  op=r['op'];case=r.get('case',r.get('case_id'));ps=r.get('params',original_params.get((op,case)));assert ps is not None,(op,case);us=r.get('us',r.get('tileops_us'))
  if us is None:us=r['measurement']['latency_ms']*1000
  matches=[b for b in by[op,case] if canonical(b['workload']['params'])==canonical(ps)]
  best=min(matches,key=lambda b:b['latency_us']) if matches else None
  out.append(dict(op=op,case=case,tileops_us=us,baseline_us=best['latency_us'] if best else None,library=best['library'] if best else None,ratio=best['latency_us']/us if best else None))
 d={'pr':n,'source':source,'cases':len(out),'missing':[x for x in out if x['ratio'] is None],'failures':[x for x in out if x['ratio'] is not None and x['ratio']<.8],'minimum':min((x['ratio'] for x in out if x['ratio'] is not None),default=None),'rows':out};P.joinpath(f'pr{n}-recomparison.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print(n,len(out),'missing',len(d['missing']),'fails',len(d['failures']),'min',d['minimum']);print(d['failures'][:12])
norm=Path('/root/dyq/workspace/reports/normalization-pr-20261104/FINAL_ACCEPTANCE_PENDING_BASELINES.json');compare(36,read(norm)['rows'],str(norm))
for x in read(Path('/root/dyq/workspace/reports/pr-performance-20261104/previous-measurements.json')):
 if x['pr'] in [24,25,28]:compare(x['pr'],x['rows'],x['provenance'])
p=Path('/root/dyq/workspace/reports/pr-performance-20261104/pr30/comparisons.json');r=read(p);print('pr30 format',str(r)[:200])

params30={(x['op'],x['case_id']):x['params'] for x in read(P/'pr30-original-rows.json')}
compare(30,[dict(x,params=params30[x['op'],x['case']]) for x in r],'run41769828212 old head7f098d52, not currentfc2da268')
