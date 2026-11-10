"""Record this completed inspection; do not acknowledge mere alarm delivery."""
import json, time, fcntl, os, tempfile, datetime
from pathlib import Path
R=Path('reports/baseline-cache-20261027').resolve();D=R/'inspection-20261028T121544'
def read(p):return json.loads(p.read_text())
def atomic(path,data):
    fd,name=tempfile.mkstemp(prefix=path.name+'.inspection-',dir=path.parent)
    with os.fdopen(fd,'w') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(name,path)
snap={h:read(sorted(D.glob(h+'-followup-*.json'))[-1]) for h in ('local','server2')}
reloads={h:read(D/(h+'-reload-state.json')) for h in snap}
for h,s in snap.items():
    assert reloads[h]['phase']=='reloaded', (h,reloads[h])
    assert s['native']['supervisor']['matches'] and s['native']['supervisor']['state']!='Z'
    assert s['held_on_disk']==s['held_added']==s['held_in_live'],(h,s['held_on_disk'],s['held_in_live'])
    assert not s['stop']
audits=[]
for name in ['AUDIT_LOCAL_20261028T1217.json','AUDIT_SERVER2_20261028T1217.json','AUDIT_LOCAL_DELTA_20261028T1236.json','AUDIT_SERVER2_DELTA_20261028T1236.json']:
    a=read(D/name);assert a['rejected']==0
    audits.append(dict(path=str(D/name),host=a['host'],checked=a['checked'],usable=a['usable'],rejected=a['rejected'],fresh_hashes=a['fresh_hashes'],device_runs=a['device_runs']))
now=time.time();utc=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat()
report=dict(inspection_id='2026-10-28T12:15:44.214085+00:00',alarm_id='768df9ce-f7d8-4a92-9ed6-5328f928958c',
    checked_at=now,checked_at_utc=utc,kind='four_hour_cache_inspection',assignment_revision=5,
    hosts=snap,cache_total=sum(s['cached'] for s in snap.values()),audit_usable=sum(a['usable'] for a in audits),audit_rejected=0,audits=audits,
    accepted_last_1h=sum(s['cached_last_1h'] for s in snap.values()),accepted_last_4h=sum(s['cached_last_4h'] for s in snap.values()),
    clock_note='remote2 clock is approximately94.5s behind local; rolling windows use each host own clock',
    initial_sample=dict(local_cached=1235,remote_cached=1128,ready_local=1251,ready_remote=1226,cached_last_hour=92,cached_last_four_hours=342),
    actions=['31 exact unrun default FP16 GDN interface mismatch pairs held; successful exact override routes retained',
             '85 exact remote missing-provider pairs held after current registry, exact registration and historical fallback inspection',
             'Fixed CPU supervisor stale QUARANTINE snapshot overwrite with shared-lock delta merge; controller changes and deletions preserved',
             'Activated patched supervisors at normal pair boundaries; no running pair or foreign workload killed',
             'No baseline kernel/compiler/schedule/checker/timing protocol edits; no cached pair reruns; no remote1 access'],
    reloads=reloads,problem_ranking=str(D/'PROBLEM_RANKING.json'),patch=str(D/'QUARANTINE_MERGE_FIX.diff'),tests=str(D/'QUARANTINE_MERGE_CHECKS.json'),
    effect_note='Script consumption and subsequent actual cache growth are recorded per host. No causal throughput improvement percentage claimed from this short observation.',
    remaining_limitations=['Local large FP8 composite routes can pass correctness but exceed600s full profiling; only exact attempts held, never count partial profiles as cache',
                           'Original TLA compiler errors/native crashes retained; no broad unsupported inference or unchanged retries',
                           'Fresh-hash audits cover frozen prefixes plus non-overlapping delta; later cache growth is separately reported'],
    next_check='2026-10-28T16:15:44.214085+00:00')
path=D/'RESULT.json';atomic(path,report)
with (R/'INSPECTIONS.jsonl').open('a') as f:f.write(json.dumps(report,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
atomic(R/'LAST_INSPECTION.json',dict(report=str(path),**report))
with (R/'RECURRING_ALARMS.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);a=read(R/'RECURRING_ALARMS.json')
    nodes=[n for n in a['milestones'] if n.get('alarm_id')==report['alarm_id']];assert len(nodes)==1
    nodes[0].update(checked_at=now,inspection_report=str(path))
    assert any(abs(n['due_at']-1793204144.2140849)<.01 for n in a['milestones'])
    atomic(R/'RECURRING_ALARMS.json',a)
m=Path('.agent_memory/THREADS/baseline-inspector-20261023')
with (m/'.state.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);d=read(m/'STATE.json');old=d['baseline_cache_20261027']
    with (D/'owned-memory-before.json').open('x') as f:json.dump(old,f,ensure_ascii=False,indent=2)
    old.update(updated_at=utc,status='both_hosts_running_after_scoped_holds_and_quarantine_merge_repair',last_inspection=dict(report=str(path),checked_at=now,checked_at_utc=utc,cache_total=report['cache_total'],audit_usable=report['audit_usable'],audit_rejected=0,next_check=report['next_check']),
               recurring_next_due_at=report['next_check'],active_followup='Continue both original CPU supervisors with exact holds. No remote1 or old workers. Next4h idle-only node16:15UTC; unchanged original clock.',
               local_supervisor_pid_at_restart=reloads['local']['supervisor']['pid'],remote_supervisor_identity=reloads['server2']['supervisor'],
               next_actions=['Inspect actual cache growth, rolling1h/4h throughput and native PID/start/boot on both permitted hosts',
                             'Measure current executable remainder; initial parity is not historical total assignment parity',
                             'Preserve exact holds and failures; RETRY only after real changed condition, never valid caches',
                             'Use QUARANTINE.lock for controller edits; patched supervisor merges only automatic deltas, never stale snapshots'],
               quarantine_update_lesson='Observed an in-flight FP8 timeout overwrite31 newly added controller holds. Both CPU supervisors patched to merge newly generated automatic changes under QUARANTINE.lock, preserving newer control changes/deletions. Reload only at normal pair boundary and verify live consumption; writing file alone is not effect.',
               current_ready={h:s['ready'] for h,s in snap.items()},current_cached={h:s['cached'] for h,s in snap.items()})
    atomic(m/'STATE.json',d)
print(json.dumps({k:report[k] for k in ['checked_at_utc','cache_total','audit_usable','accepted_last_1h','accepted_last_4h','next_check']},ensure_ascii=False))
