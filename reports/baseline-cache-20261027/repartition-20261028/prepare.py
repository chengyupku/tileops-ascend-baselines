"""CPU-only stage of a user-authorized host repartition. Does not publish it."""
import collections
import copy
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/dyq/workspace')
R = ROOT / 'reports/baseline-cache-20261027'
D = R / 'repartition-20261028'
read = lambda p: json.loads(p.read_text())
old = read(R / 'PLAN.json')
inventories = {h: read(D / (h + '-inventory.json')) for h in ('local', 'server2')}
for h, inv in inventories.items():
    assert inv['live']['phase'] == 'stopped_by_control', h
cached = {k: h for h, inv in inventories.items() for k in inv['cache_ids']}
assert len(cached) == sum(len(i['cache_ids']) for i in inventories.values())
assert len({t['id'] for t in old['pairs']}) == 6419
remote = inventories['server2']
q = remote['quarantine']
plan = copy.deepcopy(old)
local_q = read(R / 'local/QUARANTINE.json')
enabled = inventories['local']['ops_math_enabled_cases']
changes, new_holds = [], {}
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
whole = {'ops-math','tilelang-ascend','DeepGEMM-Ascend','DeepEP-Ascend','FlashMLA','DeepSelect'}
for task in plan['pairs']:
    ident = task['id']; lib, op, case = task['key']
    if ident in cached:
        assert task['host'] == cached[ident], 'Never move valid cached pairs'
        continue
    if task['host'] != 'server2':
        continue
    scope = lib + '::' + op
    if lib not in whole and lib not in q and scope not in q:
        continue
    prior = remote['latest'].get(ident, {})
    # Preserve exact semantic/numerical failures; another host is not a fix.
    if lib not in whole and prior.get('status') not in (None, 'temporarily_skipped_library', 'resource_wait'):
        error = str(prior.get('error', ''))
        if not any(s in error for s in ('deployment_blocked', 'No module named', 'No such file',
                                         'Missing library', 'ASCEND_CUSTOM_OPP_PATH', 'No registered provider')):
            continue
    reason = 'Remote deployment unavailable; user assigned missing deployments to local, preserving original case and implementation'
    task.update(host='local', previous_host='server2', assignment_revision=1,
                assignment_changed_at=stamp, assignment_reason=reason)
    hold = None
    if lib == 'FlashMLA':
        hold = 'Both hosts fail original build/source provenance check; migration cannot repair the stale build. Do not rewrite hashes.'
    elif lib == 'catlass':
        hold = 'Original CATLASS vector/MLIR Python native bindings absent on both hosts; retained native routes continue on server2.'
    elif lib == 'ops-math':
        if op not in ('AddFwdOp', 'FFTC2CFwdOp'):
            if case not in enabled.get(op, []):
                hold = 'Local public registration does not enable this exact case. Candidate evidence is not runtime registration; no forced promotion.'
            else:
                hold = 'Existing original route uses a foreground subprocess; current unified parent profiler explicitly rejects incomplete child-kernel coverage. Do not publish partial timings.'
        elif op == 'AddFwdOp' and case not in enabled.get(op, []):
            hold = 'Local public Add registration does not enable this exact case.'
    elif lib == 'vllm-ascend' and op != 'DeltaNetFwdOp':
        hold = 'Both hosts lack the source-owned native extension at the registered build path; a separately installed binary cannot bypass the original ownership guard.'
    if hold:
        new_holds['pair:' + ident] = {'reason':hold,'host':'local','key':task['key'],
            'recorded_at':stamp,'repartition_revision':1,'temporary':True,
            'evidence':'reports/baseline-cache-20261027/repartition-20261028/local-native-readiness.json; original provider and profiler source inspected'}
    changes.append({'id':ident,'key':task['key'],'from':'server2','to':'local',
                    'previous_status':prior.get('status','unprocessed'),
                    'new_condition':reason,'hold':hold})

# Put one first unmeasured pair per moved, runnable library in front, so every
# destination deployment is exercised early. Valid caches are never rerun.
first, seen = [], set()
changed = {x['id']: x for x in changes}
for t in plan['pairs']:
    if t['id'] in changed and not changed[t['id']]['hold'] and t['key'][0] not in seen:
        seen.add(t['key'][0]); first.append(t)
first_ids = {t['id'] for t in first}
plan['pairs'] = first + [t for t in plan['pairs'] if t['id'] not in first_ids]
plan['counts'] = dict(collections.Counter(t['host'] for t in plan['pairs']))
plan['assignment_revision'] = 1
plan['assignment_revision_at'] = stamp
plan['previous_plan_sha256'] = hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest()
plan['assignment_evidence'] = 'repartition-20261028/ASSIGNMENT_CHANGES.json'
old_by_id = {t['id']:t for t in old['pairs']}
for task in plan['pairs']:
    prior = old_by_id[task['id']]
    assert task['key'] == prior['key'] and task['case'] == prior['case']
    assert task['timeout'] == prior['timeout']
    if task['id'] in cached: assert task == prior
assert set(old_by_id) == {t['id'] for t in plan['pairs']}
local_q.update(new_holds)
campaign = read(R/'CAMPAIGN.json')
campaign['initial_assignments'] = campaign.get('initial_assignments', campaign['assignments'])
campaign.update(assignments=plan['counts'], assignment_revision=1, assignment_updated_at=stamp)
summary = {'at':stamp,'revision':1,'scope':6419,'moved':len(changes),
           'moved_by_library':dict(collections.Counter(x['key'][0] for x in changes)),
           'destination_ready_for_full_check':sum(x['hold'] is None for x in changes),
           'destination_held':len(new_holds),'new_assignments':plan['counts'],
           'cached_preserved':len(cached),'cached_by_host':{h:len(i['cache_ids']) for h,i in inventories.items()},
           'first_destination_probes':[t['key'] for t in first],
           'changes':changes}
for name, value in [('PLAN.proposed.json',plan),('LOCAL_QUARANTINE.proposed.json',local_q),
                    ('CAMPAIGN.proposed.json',campaign),('ASSIGNMENT_CHANGES.json',summary)]:
    with (D/name).open('x') as f: f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='changes'},ensure_ascii=False))
