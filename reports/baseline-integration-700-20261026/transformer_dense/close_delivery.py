import json,datetime,pathlib,collections
p=pathlib.Path(__file__).parent
r=json.loads((p/'RESULT.json').read_text());q=json.loads((p/'QUEUE_ACCOUNTING.json').read_text());mt=json.loads((p/'GUARD_EXIT_MTIMES.json').read_text());byticket={x['ticket']:x for x in q['rows']}
for row in r['rows']:
 if row['status']!='completed':continue
 f=p/pathlib.Path(row['checker_receipt']).parent/'guard_snapshot.json'
 if not f.exists():continue
 s=json.loads(f.read_text());g=row['guard_receipt'];end=mt.get(pathlib.Path(g).with_suffix('.exit').name)
 if end:
  upper=end-datetime.datetime.fromisoformat(s['acquired_at']).timestamp()
  assert upper>0
  row['timings']['guard_hold_upper_bound_s']=upper
  item=byticket.get(s.get('queue_ticket'))
  if item:item.update(hold_upper_bound_s=upper,guard_exit_receipt=pathlib.Path(g).with_suffix('.exit').name)
q['held_time_note']='hold_upper_bound_s uses same-host acquired_at through shell exit receipt mtime, including guard cleanup and receipt-write overhead; exact device/kernel time remains null.'
(p/'QUEUE_ACCOUNTING.json').write_text(json.dumps(q,indent=2));(p/'RESULT.tmp').write_text(json.dumps(r,indent=2));(p/'RESULT.tmp').replace(p/'RESULT.json')
d=json.loads((p/'DELIVERY.json').read_text());d['unified_patches']=sorted(x.name for x in p.glob('provider_init*.unified.diff'));d['timings'].update(measured_unique_guard_batches=q['measured_unique_batches'],measured_queue_s=q['measured_queue_s'],queue_accounting='QUEUE_ACCOUNTING.json',active_cpu_s=None,held_time_note=q['held_time_note']);d['checks'].append('QUEUE_ACCOUNTING.json');d['final_verification']='39 complete numerical checks and 39 exact formal provider routes passed; route checks were device-free.';now=datetime.datetime.now(datetime.timezone.utc).isoformat();d['utc']=now;d['actual_elapsed_wall_s']=(datetime.datetime.fromisoformat(now)-datetime.datetime.fromisoformat(d['started_at'])).total_seconds();(p/'DELIVERY.tmp').write_text(json.dumps(d,indent=2));(p/'DELIVERY.tmp').replace(p/'DELIVERY.json')
(p/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='completed_with_blockers',current_keys=[],last_progress_at=now,blocked_reason='39 completed and formally registered; 93 blocked by evidenced API/common-route incompatibilities; 2 canonical numerical failures.'),indent=2))
with (p/'REVIEW.md').open('a') as f:f.write('\nFinal delivery: corrected scope 134 keys; 39 completed and formally registered, 93 blocked, 2 failed. INT8 GEMM 6/6, FP8 BMM 11/11, FP8 grouped MoE 14/14; eight additional graph/extension keys passed. Final device-free exact-route check passed all 39. Detailed immutable receipts and registration backups are indexed in RESULT.json and DELIVERY.json.\n')
print(json.dumps(dict(status_counts=r['status_counts'],elapsed_wall_s=d['actual_elapsed_wall_s'],measured_queue_s=q['measured_queue_s'])))
