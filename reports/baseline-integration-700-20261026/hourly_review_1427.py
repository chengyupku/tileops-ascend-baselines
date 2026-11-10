import pathlib,json,datetime,time,sqlite3,collections,csv,fcntl,subprocess,hashlib
D=pathlib.Path(__file__).resolve().parent;now=time.time();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();state=json.loads((D/'STATE.json').read_text());plan=json.loads((D/'PLAN.json').read_text());jobs=json.loads((D/'JOBS.json').read_text())['jobs'];accepted=json.loads((D/'ACCEPTED.json').read_text())['rows'];start=datetime.datetime.fromisoformat(state['started_at']).timestamp();boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();con=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
activities=[];identities=[];states=[]
for j in jobs:
 st,data=con.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone();n=json.loads(data);states.append(st);end=n.get('finished_at',now);wd=pathlib.Path(j['directory']);counts={}
 if (wd/'RESULT.json').exists():
  z=json.loads((wd/'RESULT.json').read_text());rr=z.get('rows',[]) if isinstance(z,dict) else z;counts=dict(collections.Counter(r.get('status','unknown') for r in rr))
 acc=0
 for r in accepted:
  cp=r.get('checker','');cp=cp.get('path','') if isinstance(cp,dict) else cp
  if cp.startswith(str(wd)+'/'):acc+=1
 activities.append(dict(worker=j['name'],job=n['id'],host=j['host'],role=j.get('role','integration'),native_status=st,wall_minutes=round((end-n['created_at'])/60,2),scope=j['count'],reported_completed=counts.get('completed',0),reported_private_pass=counts.get('verified_unregistered',0),parent_accepted_from_this_worker_receipts=acc,reported_failed=counts.get('failed',0),reported_blocked=counts.get('blocked',0),model=n.get('model'),effort=n.get('effort')))
 if st in ['running','starting']:
  alive=False;actual=None
  try:
   f=pathlib.Path('/proc')/str(n['child_pid'])/'stat';ss=f.read_text().rsplit(')',1)[1].split();actual=boot+':'+ss[19];alive=ss[0]!='Z' and actual==n.get('child_stamp')
  except (OSError,KeyError,IndexError):pass
  identities.append(dict(worker=j['name'],pid=n.get('child_pid'),expected_stamp=n.get('child_stamp'),actual_stamp=actual,alive=alive,model=n.get('model'),effort=n.get('effort')))
with (D/'HOURLY_AGENT_ACTIVITY_20261026T1427.csv').open('w') as f:
 cw=csv.DictWriter(f,fieldnames=list(activities[0]));cw.writeheader();cw.writerows(activities)
# Time-weighted occupied slots, separate from useful work or NPU utilization.
sn=[]
for line in (D/'SNAPSHOTS.jsonl').open():
 d=json.loads(line);t=datetime.datetime.fromisoformat(d['at']).timestamp()
 if start<=t<=now:sn.append((t,d['active_workers'],d['live_active_workers']))
sn.sort();area=livearea=0
for i,(t,a,l) in enumerate(sn):
 dt=(sn[i+1][0] if i+1<len(sn) else now)-t;area+=a*dt;livearea+=l*dt
metrics=dict(accepted=len(accepted),remaining=700-len(accepted),elapsed_minutes=(now-start)/60,accepted_per_hour=len(accepted)*3600/(now-start),first_hour_accepted=sum(datetime.datetime.fromisoformat(r['at']).timestamp()<=start+3600 for r in accepted),active_workers=len(identities),live_workers=sum(r['alive'] for r in identities),live_survival=sum(r['alive'] for r in identities)/len(identities),time_weighted_slot_occupancy=area/(8*(now-sn[0][0])),time_weighted_observed_survival=livearea/area,productive_utilization=None,api_scheduling_queue_seconds=None,native_status_counts=dict(collections.Counter(states)))
# Scope accounting includes completed workers' numeric passes, even if parent acceptance is still pending.
key=lambda r:tuple(r.get('key') or [r['baseline'],r['operator'],r['case_id']]);owners={};groups={}
def add(name,rr):
 groups[name]=len(rr)
 for r in rr:owners.setdefault(key(r),[]).append(name)
for w in plan['workers']:
 if w.get('active_assignment',True):
  x=json.loads((pathlib.Path(w['directory'])/'SCOPE.json').read_text());add(w['name'],x.get('rows',[]) if isinstance(x,dict) else x)
for f in state['root_queued_scopes']:
 x=json.loads(pathlib.Path(f).read_text());add(pathlib.Path(f).parent.name if pathlib.Path(f).name=='SCOPE.json' else pathlib.Path(f).name,x.get('rows',[]) if isinstance(x,dict) else x)
for name in state['closed_success_scopes']:
 x=json.loads((D/name/'RESULT.json').read_text())['rows'];add(name+'_closed_success',[r for r in x if r['status'] in ['completed','verified_unregistered']])
assert len(owners)==700 and all(len(v)==1 for v in owners.values()),groups
ownership=dict(at=stamp,total_unique=700,execution_scope=328,root_queued=221,closed_worker_success_under_review_or_accepted=151,duplicates=0,groups=groups)
# Parent review terminal deliveries. Numeric receipts awaiting independent auditor stay unaccepted.
(D/'tla_simple_tilelang/PARENT_REVIEW.json').write_text(json.dumps(dict(at=stamp,scope=80,reported_registered=44,parent_accepted=23,remaining_successes_under_audit03=21,remaining_not_completed=36,wall_seconds=4144.756943,reservation_queue_seconds=1741.185744,duplicate_verified_case_receipts=0,additional_repair_attempts=8,guard_accounting_correction=str(D/'PARENT_TLA5_REVIEW.json'),source_and_frozen_evidence='Remaining21 assigned independent CPU audit03; CLI completed not acceptance.'),indent=2))
(D/'nn_recovery01/PARENT_REVIEW.json').write_text(json.dumps(dict(at=stamp,scope=48,private_verified=21,parent_now_registered_accepted=21,numeric_failed=2,blocked=25,next_assignment='23 interrupted/unexecuted keys now nn_recovery02;4 otherfailures retainedrootqueue',evidence=[str(D/'PARENT_AUDIT_1418_REVIEW.json'),str(D/'nn_parent_last3/REGISTRATION.json')],queue_and_idle_seconds=263.540174,wall_seconds=1577.891755),indent=2))
(D/'sgl_gdn_recovery02/PARENT_REVIEW.json').write_text(json.dumps(dict(at=stamp,scope=38,accepted=0,probe_count=1,actual_api='chunk_gated_delta_rule_native',unresolved='Devicefailure duringFP64 diagnostic; no valid fullchecker result. No generalized numericverdict for37unexecuted.',next_assignment='sgl_gdn_recovery03 private diagnostic-onlyCPUcandidate; originalvalidator/math/precision/tolerance unchanged, explicitoverrideprovenance required',wall_seconds=570.042053,device_wait_including_idle_seconds=255.41908,device_command_seconds=48.741589),indent=2))
services=subprocess.run(['systemctl','show','baseline700-20261026-hourly.timer','baseline700-20261026-hourly.service','baseline700-20261026-sample.timer','baseline700-20261026-sample.service','-p','Id','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','ExecMainExitTimestamp'],capture_output=True,text=True).stdout
report=dict(at=stamp,scheduled_snapshot_at='2026-10-26T14:11:36.777717+00:00',metrics=metrics,process_identity=identities,service_health=services,next_hourly_utc='2026-10-26T15:11:36+00:00',host_status_evidence=[str(D/n) for n in ['HOURLY_1415_LOCAL_STATUS.json','HOURLY_1415_SERVER1_STATUS.json','HOURLY_1415_SERVER2_STATUS.json']],all3_hosts_health_code=0,guard_queue_samples={'local_reservation_wait_s':0.0004,'local_physical_idle_wait_s':255.41868,'server1_reservation_wait_s':27.01697,'server2_reservation_wait_s':84.127961},queue_interpretation='Host clocks differ93s; durations only within host. Local acquired reservation immediately but waitedphysicalidle255s. TLA aggregate queue29min/69;linear queue+startup24.3min/55 is not pure queue.',largest_common_issues=[{'issue':'Evidence acceptance/registration backlog','action':'Accepted35 more duringhourlyreview:NN18+GDN9+W4last3+TLA5. Current146. CPUaudit03assigned41newkeys; audit02finishesremaining andsourcechangedcase.'},{'issue':'Shared device waiting plus blockedoriginaldeployments','evidence':'TLA29minreservationqueue,linear24.3minqueue+startup; scope28FP16/Delta/Causal originaldeployment gaps transferredoneworker server2.'},{'issue':'Workers not using intended runtime/dispatch entrypoint','evidence':'SGLGDN38 unusedfactory andSGLconv9 wrongsystempython; hosttorchexists inbaseline-run CPUpreflight','action':'Rejectedfalseblockers, replacedwithmediumworkers, providedfixedRUN_GUARDED.sh plusactualdispatch preflight. No systemreinstallation.'},{'issue':'Avoidable rerun/setup onNPU','evidence':'Attention1fullpassedMLA rerunonlybecauseunrelatedregistrationchanged;vllm2deviceprobes failedmissingmodule/config','action':'ScopedfeedbackrequiresCPUdependencies/preflight, preserveeachattempt, useCPU sourceequivalence audit insteadofnumericreruns.'},{'issue':'Childchecker0 mislabeled as guard0','action':'Preserve true batch139/124 and separate earlier synchronizedcase evidence; no extra NPU rerun.'}],agent_activity_csv=str(D/'HOURLY_AGENT_ACTIVITY_20261026T1427.csv'),activities=activities,ownership=ownership,remaining_limits='No lowprocesssurvival observed. CompletedCLIzero-success workers are task-quality/blocker cases, not processdeaths. Slotoccupancy not productiveutilization; APIqueue unavailable. OldHOLDqueues remaininactive.')
fn=D/'INSPECTION_20261026T1427.json';fn.write_text(json.dumps(report,indent=2,ensure_ascii=False));state.update(completed=len(accepted),parent_accepted_completed=len(accepted),active_workers=len(identities),observed_active_survival=metrics['live_survival'],last_inspection=stamp,last_inspection_report=str(fn),next_hourly_inspection_utc=report['next_hourly_utc'],ownership_audit=ownership,last_status_at=stamp,last_status_accepted=len(accepted));(D/'STATE.json').write_text(json.dumps(state,indent=2))
m=D.parents[1]/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(m)+'.lock','a') as l1,open(m.parent/'STATE.lock','a') as l2:
 fcntl.flock(l1,fcntl.LOCK_EX);fcntl.flock(l2,fcntl.LOCK_EX);d=json.loads(m.read_text());x=d['integration_700_20261026'];x.update(parent_accepted_completed=len(accepted),last_inspection=stamp,last_inspection_report=str(fn),next_hourly_inspection_utc=report['next_hourly_utc'],root_queued_scopes=state['root_queued_scopes'],root_queued_count=state['root_queued_count'],ownership_audit=ownership,active_workers=len(identities),last_status_at=stamp,closed_success_scopes=state['closed_success_scopes'])
 for k,v in state.items():
  if k.endswith('_job'):x[k]=v
 x['hourly_metrics']=metrics;tmp=m.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False));tmp.replace(m)
print(json.dumps(dict(report=str(fn),metrics=metrics,ownership=ownership),ensure_ascii=False))
