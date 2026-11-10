import json, math, statistics
from pathlib import Path
root=Path(__file__).resolve().parent
source=root.parent/'ci-baseline-elementwise-20261101/current-contract-baseline/BASELINE32_PRIVATE.json'
rows=json.loads(source.read_text())['rows']
def dist(xs):
 xs=sorted(xs)
 def q(p):
  n=(len(xs)-1)*p;i=int(n);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(n-i)
 return dict(n=len(xs),min=xs[0],p10=q(.1),median=q(.5),p90=q(.9),p99=q(.99),max=xs[-1]) if xs else {}
def valid(r):
 return len(r.get('end_to_end_samples_ms',[]))==15 and all(math.isfinite(x) and x>0 for x in r['end_to_end_samples_ms']) and r.get('end_to_end_mean_ms',0)>0
out={'source':str(source),'rows':len(rows),'valid_H_rows':sum(map(valid,rows)),'operators_with_H':len({r['key'][1] for r in rows if valid(r)}),'cases_with_H':len({tuple(r['key'][1:]) for r in rows if valid(r)}),'groups':{}}
for name,rs in [('single_role',[r for r in rows if 'device_sum_quarantined' not in r]),('mixed_roles',[r for r in rows if 'device_sum_quarantined' in r])]:
 ratios=[r['end_to_end_mean_ms']/r['timing']['latency_ms'] for r in rs]
 out['groups'][name]={'rows':len(rs),'H_over_v2':dist(ratios),'H_ms':dist([r['end_to_end_mean_ms'] for r in rs]),'v2_ms':dist([r['timing']['latency_ms'] for r in rs]),'H_over_v2_gt_2':sum(x>2 for x in ratios),'H_over_v2_gt_10':sum(x>10 for x in ratios),'H_over_v2_gt_100':sum(x>100 for x in ratios)}
out['limitations']=['No paired historical v3; single-role arithmetic is unchanged by role de-duplication but not a fresh v3 measurement.','Mixed-role presence does not prove same-task overlap; cannot divide v2 by two.','H measures profiled host call plus device synchronization, not device-only task spans.']
(root/'H_V2_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
