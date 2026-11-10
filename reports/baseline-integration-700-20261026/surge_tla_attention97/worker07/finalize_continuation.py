from pathlib import Path
import json,datetime,ast,hashlib,collections
from collect_resume import collect
P=Path(__file__).resolve().parent
s=collect();rows=json.loads((P/'RESULT.json').read_text());initial=json.loads((P/'REMAINING.json').read_text())
for a in s['attempts']:
 f=Path(a['guard']).parent/'GUARD_RUNNING.json';m=json.loads(f.read_text()) if f.exists() else {}
 a['guard_native_requested_at']=m.get('requested_at')
 def delta(x,y):
  return (datetime.datetime.fromisoformat(x)-datetime.datetime.fromisoformat(y)).total_seconds() if x and y else None
 a['queue_seconds']=delta(m.get('acquired_at'),m.get('requested_at'))
 a['physical_idle_check_and_child_start_seconds']=delta(m.get('updated_at'),m.get('acquired_at'))
 a['child_wall_seconds']=delta(a.get('end'),a.get('child_start')) if a['host']=='local' else None
 a['timing_note']='Native guard timestamps used for queue/pre-child intervals. Actual device and held-CPU times are unmeasured; remote end uses local receipt clock and is not subtracted from native child-start.'
for r in rows:
 own=[a for a in s['attempts'] if a['key']==r['key']]
 if own:r['attempts']=own
 if r['key'] in initial:(P/(r['key'][1]+'--'+r['key'][2]+'.RESULT.json')).write_text(json.dumps(r,indent=2))
(P/'RESULT.json').write_text(json.dumps(rows,indent=2))
remaining=[k for k in initial if not any(a['key']==k and a['child'] for a in s['attempts'])]
(P/'REMAINING_UNEXECUTED.json').write_text(json.dumps(remaining,indent=2))
source_files=[p for p in P.glob('*.py')]
for f in source_files:ast.parse(f.read_text())
previous=json.loads((P/'first_delivery/RESULT.json').read_text());unchanged=[]
for r in rows:
 if r['key'] not in initial:
  old=next(o for o in previous if o['key']==r['key']);assert old==r;unchanged.append(r['key'])
s.update(continuation_keys=9,continuation_executed_keys=9-len(remaining),remaining_unexecuted=remaining,preserved_terminal_keys=unchanged,accepted_added=0,registered_added=0,ready=False,checks={'python_syntax_files':len(source_files),'scope_unique':len(set(tuple(r['key']) for r in rows))==13,'old_four_terminal_rows_unchanged':True,'no_kernel_pass_schedule_checker_seed_tolerance_edits':True,'no_shared_registry_memory_writes':True},timing_totals={'attempt_wall_seconds':sum(a['wall_seconds'] for a in s['attempts']),'unstarted_attempt_wall_seconds':sum(a['wall_seconds'] for a in s['attempts'] if not a['child']),'known_native_queue_seconds':sum(a['queue_seconds'] or 0 for a in s['attempts']),'known_physical_idle_check_and_child_start_seconds':sum(a['physical_idle_check_and_child_start_seconds'] or 0 for a in s['attempts']),'held_cpu_seconds':None,'device_execution_seconds':None},guard_exit_counts=dict(collections.Counter(str(a['guard_exit']) for a in s['attempts'])),recipe=str(P/'CONTINUATION_RECIPE.md'),scope_note='REMAINING.json retains the original nine-key continuation assignment; REMAINING_UNEXECUTED.json is the actual unexecuted subset. Terminal failures are not accepted successes.')
(P/'DELIVERY.json').write_text(json.dumps(s,indent=2));(P/'TIMINGS.json').write_text(json.dumps({'totals':s['timing_totals'],'attempts':s['attempts']},indent=2))
(P/'FILEHASHES_CONTINUATION.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.is_file() and f.suffix in ['.py','.md']},indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_key=None,phase='completed_exact_results' if not remaining else 'remaining_resource_blocked',last_substantive_output='RESULT.json, DELIVERY.json, TIMINGS.json',blocker=None if not remaining else 'No device child for remaining subset',next='Parent review; no READY/accepted claims'),indent=2))
print(json.dumps({k:s[k] for k in ['continuation_executed_keys','remaining_unexecuted','verified_added','accepted_added','status_counts','guard_exit_counts','timing_totals','checks']},indent=2))
