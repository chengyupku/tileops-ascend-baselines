import runpy,pathlib,json,datetime,time,tempfile
P=pathlib.Path(__file__).resolve().parent;m=runpy.run_path(str(P.parent/'refill_fleet.py'));g=m['dispatch_open'].__globals__
checks=[]
with tempfile.TemporaryDirectory(dir=P,prefix='refill_cpu_check_') as tmp:
 d=pathlib.Path(tmp);g['D']=d
 def state(enabled=True,phase='running'):(d/'STATE.json').write_text(json.dumps({'enabled':enabled,'phase':phase}))
 state();assert m['dispatch_open'](time.time()+120);checks.append('enabled before deadline permits dispatch')
 state(False);assert not m['dispatch_open'](time.time()+120);checks.append('disabled blocks dispatch')
 for phase in ['paused','stopped','complete']:
  state(phase=phase);assert not m['dispatch_open'](time.time()+120);checks.append(phase+' blocks dispatch')
 state();assert not m['dispatch_open'](time.time()+30);checks.append('last minute blocks new dispatch')
 (d/'ROOT_CONTROL_QUEUE.json').write_text(json.dumps({'tasks':[{'ready':True}]}));state(False);record={'rejected':[],'launched':[]};m['dispatch_controls']({},None,time.time()+120,record);assert not record['launched'] and record['stopped'];checks.append('disabled controls never touch native DB or launch')
 for entry in [{'enabled':False},{'phase':'paused'},{'ready':False}]:assert m['eligibility'](entry,None,None,None,None,None);checks.append('ineligible entry rejected before scope reads')
(P/'REFILL_CHECKS.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':len(checks),'checks':checks,'npu_access':False},indent=2)+'\n');print('CPU refill checks passed:',len(checks))
