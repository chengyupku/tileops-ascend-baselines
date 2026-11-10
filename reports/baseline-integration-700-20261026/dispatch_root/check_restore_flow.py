"""CPU fixture only; assert no native tool can launch during this test."""
import pathlib,runpy,json,tempfile,sqlite3,datetime
P=pathlib.Path(__file__).resolve().parent;m=runpy.run_path(str(P.parent/'refill_fleet.py'));g=m['main'].__globals__;connect=sqlite3.connect;run=m['subprocess'].run;checks=[]
def write(p,x):p.write_text(json.dumps(x))
try:
 with tempfile.TemporaryDirectory(dir=P,prefix='restore_flow_cpu_') as tmp:
  d=pathlib.Path(tmp);q=d/'fixture_group';q.mkdir();w=q/'worker';w.mkdir();g['D']=d
  db=connect(':memory:');db.execute('create table jobs(id text,parent text,status text,data text)');db.executemany('insert into jobs values(?,?,?,?)',[('fixture_terminal_coordinator','fixture_root','completed','{}'),('fixture_active_child','fixture_old_thread','running','{}')]);g['sqlite3'].connect=lambda *a,**kw:db
  def forbidden(*a,**kw):raise AssertionError('Native subprocess forbidden in CPU test')
  g['subprocess'].run=forbidden
  write(d/'STATE.json',{'enabled':True,'phase':'running','deadline_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=10)).isoformat(),'parent':'fixture_only_no_launch','coordinator_groups':[q.name]});write(d/'ACCEPTED.json',{'rows':[]});write(d/'JOBS.json',{'jobs':[]});write(q/'SELF_JOB.json',{'id':'fixture_terminal_coordinator'});write(q/'CHILD_JOBS.json',{'jobs':[{'native':{'id':'fixture_active_child'},'directory':str(w)}]});write(q/'SCOPE.json',[{'key':['fixture','op','case']}]);write(w/'SCOPE.json',[{'key':['fixture','op','case']}]);write(q/'READY_QUEUE.json',{'tasks':[]});write(q/'GROUP.json',{'enabled':False})
  m['main']();assert not (d/'ROOT_CONTROL_QUEUE.json').exists();checks.append('disabled group with surviving child never triggers supervisor restart')
  write(q/'GROUP.json',{'enabled':True});m['main']();queue=json.loads((d/'ROOT_CONTROL_QUEUE.json').read_text());assert len(queue['tasks'])==1;assert json.loads((q/'CHILD_JOBS.json').read_text())['jobs'][0]['native']['id']=='fixture_active_child';checks.append('terminal coordinator with surviving child queues one supervisor and preserves child ownership')
finally:g['sqlite3'].connect=connect;g['subprocess'].run=run
(P/'RESTORE_FLOW_CHECKS.json').write_text(json.dumps({'passed':len(checks),'checks':checks,'native_launches':0,'npu_access':False},indent=2)+'\n');print('Restore flow checks passed:',len(checks))
