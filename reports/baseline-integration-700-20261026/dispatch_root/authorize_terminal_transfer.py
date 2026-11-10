import pathlib,json,sqlite3,datetime,runpy
P=pathlib.Path(__file__).resolve().parent;D=P.parent;m=runpy.run_path(str(D/'refill_fleet.py'));db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
keys={('ops-nn','GroupedGemmFwdOp',c) for c in ['moe-dgrad-float16','moe-fwd-float16']};jobs=json.loads((D/'JOBS.json').read_text())['jobs']
for name in json.loads((D/'STATE.json').read_text())['coordinator_groups']:jobs+=json.loads((D/name/'CHILD_JOBS.json').read_text())['jobs']
active_overlaps=[]
for j in jobs:
 if j.get('role')=='coordinator':continue
 r=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone()
 if not r:continue
 d=json.loads(r[1]);q=pathlib.Path(j['directory'])
 if r[0] in m['ACTIVE'] or m['live'](d.get('child_pid'),d.get('child_stamp')):
  overlap=keys & {m['key'](x) for x in json.loads((q/'SCOPE.json').read_text())}
  if overlap:active_overlaps.append({'job':j['native']['id'],'keys':list(overlap)})
assert not active_overlaps
old=db.execute('select status,data from jobs where id=?',('e2c6d57c84884b8d',)).fetchone();od=json.loads(old[1]);assert old[0]=='completed' and not m['live'](od.get('child_pid'),od.get('child_stamp'))
q=D/'threehost_nn_grouped3_build/local_fp16_01';meta=json.loads((q/'GUARD_METADATA.json').read_text());stamp=meta['boot_id']+':'+str(meta['start_ticks']);child=meta['child'];cstamp=meta['boot_id']+':'+str(child['start_ticks']);assert not m['live'](meta['pid'],stamp) and not m['live'](child['pid'],cstamp)
x={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'author':'dispatch_root 3bdcd752e1644c6d','not_an_acceptance_root_reply':True,'authority':'User authorizes necessary dispatch without parent item approval; this is terminal-owner handoff, not splitting an active owner','decision':'Grouped two exact keys may transfer to recovery_native5; acceptance root retains all result acceptance','keys':[list(k) for k in keys],'old_owner_confirmed_completion':'/root/dyq/codex-workers/jobs/e2c6d57c84884b8d/result.txt','old_owner_delivery':str(D/'threehost_nn_grouped3_build/DELIVERY.json'),'native_status':old[0],'old_child_identity_alive':False,'old_guard_identity_alive':False,'old_guard_child_identity_alive':False,'guard_ended':json.loads((q/'GUARD_RESULT.json').read_text()),'active_execution_overlaps':active_overlaps,'bmm3_decision':'Do not rerun historical complete successes; acceptance root resolves publication evidence','new_execution_requires':'Record actual changed resource/host condition first; use original guard; preserve original checker/source/seed/tolerance'}
(P/'OWNERSHIP_REPLY.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print('Documented terminal Grouped two-key transfer with real process identities and zero active execution overlap; no fabricated acceptance approval')
