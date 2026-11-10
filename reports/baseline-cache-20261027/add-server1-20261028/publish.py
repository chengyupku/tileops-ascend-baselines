import json,hashlib,os,sys,collections
from pathlib import Path
host=sys.argv[1];assert host in ('local','server1','server2')
R=Path('reports/baseline-cache-20261027');D=R/'add-server1-20261028';H=R/host
new=(D/'PLAN.proposed.json').read_bytes();plan=json.loads(new);assert len(plan['pairs'])==len({t['id'] for t in plan['pairs']})==6419
changes=json.loads((D/'ASSIGNMENT_CHANGES.json').read_text())['changes'];changed={x['id'] for x in changes}
if host!='server1':
    assert (H/'STOP').read_text()=='add-server1-20261028/ownership-boundary\n'
    old=(R/'PLAN.json').read_bytes();assert hashlib.sha256(old).hexdigest()==plan['previous_plan_sha256']
    before={t['id']:t for t in json.loads(old)['pairs']};after={t['id']:t for t in plan['pairs']}
    assert all({k:v for k,v in t.items() if k!='host'}=={k:v for k,v in after[i].items() if k!='host'} for i,t in before.items())
    cache=[json.loads(x) for x in (H/'CACHE.jsonl').read_text().splitlines()]
    assert all(after[c['task_id']]['host']==host for c in cache if before[c['task_id']]['host']==host)
    live=json.loads((H/'LIVE.json').read_text())
    assert live.get('key') not in [x['key'] for x in changes if x['from']==host]
    with (D/(host+'-PLAN.before.json')).open('xb') as f:f.write(old)
    campaign=json.loads((R/'CAMPAIGN.json').read_text())
else:
    assert not (R/'PLAN.json').exists()
    campaign=json.loads((D/'CAMPAIGN.seed.json').read_text());H.mkdir(exist_ok=True)
    for name,value in [('journal.jsonl',''),('CACHE.jsonl',''),('QUARANTINE.json','{}\n'),('RETRY.json','{}\n')]:
        with (H/name).open('x') as f:f.write(value)
campaign.update(hosts={'local':'/root/dyq/workspace','server1':'openlibing-DevEnv-661495:/opt/tileops-baseline-cache/workspace','server2':'openlibing-DevEnv-055409:/opt/tileops-all-pairs/workspace'},no_server1=False,
                server1_authorization='User20261028 released server1 and requested three cards in parallel',assignments=plan['counts'],assignment_revision=plan['assignment_revision'],assignment_updated_at=plan['assignment_updated_at'])
for name,data in [('PLAN.json',new),('CAMPAIGN.json',(json.dumps(campaign,ensure_ascii=False,indent=2)+'\n').encode())]:
    tmp=R/(name+'.threehost-tmp');tmp.write_bytes(data);os.replace(tmp,R/name)
print(json.dumps({'host':host,'revision':plan['assignment_revision'],'sha256':hashlib.sha256(new).hexdigest(),'counts':plan['counts']}))
