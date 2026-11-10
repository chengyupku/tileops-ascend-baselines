import collections, datetime, hashlib, json, pathlib
R=pathlib.Path('reports/baseline-cache-20261027');D=R/'inspection-20261029T001544'
read=lambda p:json.loads(p.read_text())
old=(R/'PLAN.json').read_bytes();plan=json.loads(old)
assert plan['assignment_revision']==6
snap={h:read(D/(h+'-BOUNDARY.json')) for h in ('local','server1','server2')}
assert all(s['supervisor_lock_free'] and not s['prior_identity_alive'] for s in snap.values())
assert not snap['local']['ready'] and not snap['server1']['ready']
ready=snap['server2']['ready'];tasks={t['id']:t for t in plan['pairs']}
proof={h:{t['id']:t for t in read(D/(h+'-MIGRATION_PREFLIGHT.json'))['tasks']} for h in snap}
def weight(t):
    op=t['key'][1]
    return {'DSADecodeWithKVCacheFwdOp':441,'MLADecodeWithKVCacheFwdOp':350,
        'GDNFwdOp':300 if t['key'][0]=='sgl-kernel-npu' else 400,'KDAFwdOp':300,
        'GQADenseFwdOp':240,'GQAPagedFwdOp':240,'MoEGroupedGemmFwdOp':180,'MoEExpertMLPFwdOp':180}.get(op,120)
def allowed(t):
    a=['server2']
    if t['key'][0]!='sgl-kernel-npu':return a
    for h in ['local','server1']:
        p=proof[h][t['id']]
        if p['cached'] or p['hold'] or any(s not in ('temporarily_skipped_library','resource_wait') for s in p['prior_statuses']):continue
        if p['route_observation']!='CPU_ONLY_BLOCK:torch':continue
        a.append(h)
    return a
loads={h:0 for h in snap};chosen={}
for t in sorted(ready,key=lambda x:(len(allowed(x)),-weight(x),x['id'])):
    dest=min(allowed(t),key=lambda h:(loads[h],h))
    chosen[t['id']]=dest;loads[dest]+=weight(t)
changes=[]
for t in ready:
    dest=chosen[t['id']]
    assert t['id'] not in set().union(*(set(s['all_cache_ids']) for s in snap.values()))
    if dest!='server2':
        changes.append({'id':t['id'],'key':t['key'],'from':'server2','to':dest,'estimated_attempt_seconds':weight(t),
            'basis':'Unattempted at destination; no holds/cache; same generic CPU-resolved SGL adapter route; original native library already audited on destination.'})
        tasks[t['id']]['host']=dest
counts=dict(collections.Counter(t['host'] for t in plan['pairs']))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
plan.update(assignment_revision=7,previous_plan_sha256=hashlib.sha256(old).hexdigest(),counts=counts,
    assignment_updated_at=now,assignment_evidence='inspection-20261029T001544/ASSIGNMENT_CHANGES.json',
    assignment_reason='Two drained cards; redistribute only genuinely unattempted ready SGL work by coarse wall-time estimate; keep vLLM on its established server2 deployment and preserve all cached ownership.')
result={'at':now,'changes':changes,'ready_before':len(ready),
    'ready_after':dict(collections.Counter(chosen.values())), 'estimated_work_seconds':loads,
    'weights_are': 'coarse scheduling estimates informed by recent long-case costs, not an ETA guarantee',
    'no_moved_cached_or_numerically_failed_pairs':True,'no_provider_or_runtime_change':True}
for name,obj in [('PLAN.revision7.json',plan),('ASSIGNMENT_CHANGES.json',result)]:
    with (D/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='changes'},ensure_ascii=False));print('new_total_ownership',counts)
