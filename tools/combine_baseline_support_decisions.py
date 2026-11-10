"""CPU-only two-bucket ledger for this conversation's explicitly owned pairs."""
import collections
import csv
import fcntl
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]
r=ROOT/'reports/baseline-cache-20261027/support-run-20261029'
scope={x['id'] for x in json.load(open(r/'SCOPE.json'))}
policy=json.load(open(r/'POLICY.json'))
assert len(scope)==policy['scope']==1738
records={}; host_snapshots={}
for name in ['MANUAL_DECISIONS.json','SUPPORTED_EXISTING_CORRECTNESS.json']:
    for row in json.load(open(r/name)):
        assert row['id'] in scope
        records[row['id']]=row
for host in policy['allowed_execution_hosts']:
    d=json.load(open(r/(host+'-DECISIONS.json')))
    host_snapshots[host]={'at':d['at'],'assignment_revision':d['assignment_revision'],'supported':len(d['supported']),'cached':d['cached']}
    for row in d['supported']:
        assert row['id'] in scope
        prior=records.get(row['id'])
        if prior and prior['bucket'] != row['bucket']:
            raise ValueError('Conflicting supported and unsupported evidence for '+row['id'])
        if prior and prior.get('performance_cached') and not row.get('performance_cached'):
            raise ValueError('Refusing to downgrade cached evidence for '+row['id'])
        records[row['id']]=row
counts=collections.Counter(row['bucket'] for row in records.values())
summary={'at':time.time(),'scope':len(scope),'classified':len(records),
         'baseline支持':counts['baseline支持'],'baseline不支持':counts['baseline不支持'],
         'performance_cached_in_scope':sum(bool(x.get('performance_cached')) for x in records.values()),
         'remaining_to_classify':len(scope)-len(records),'old1076_delegated':True,
         'allowed_hosts':policy['allowed_execution_hosts'],'host_snapshots':host_snapshots}
with (r/'CLASSIFICATION.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    p=r/'CLASSIFICATION.json';out={'summary':summary,'decisions':list(records.values())}
    if p.exists():
        (r/('CLASSIFICATION.before-'+str(time.time_ns())+'.json')).write_bytes(p.read_bytes())
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
    p=r/'CLASSIFICATION.csv';tmp=p.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','library','op','case','host','bucket','performance_cached','evidence'])
        for ident,row in sorted(records.items()):
            writer.writerow([ident,*row['key'],row.get('host',''),row['bucket'],bool(row.get('performance_cached')),row.get('correctness_receipt',row.get('evidence',''))])
    tmp.replace(p)
print(json.dumps(summary,ensure_ascii=False))
