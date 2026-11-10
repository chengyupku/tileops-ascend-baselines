import runpy,pathlib,json,tempfile,sqlite3,time
P=pathlib.Path(__file__).resolve().parent;m=runpy.run_path(str(P.parent/'refill_fleet.py'));g=m['restore_supervision'].__globals__
with tempfile.TemporaryDirectory(dir=P,prefix='supervisor_cpu_check_') as tmp:
 d=pathlib.Path(tmp);g['D']=d;p=d/'group';p.mkdir();(p/'GROUP.json').write_text('{}');db=sqlite3.connect(':memory:');db.execute('create table jobs(id text,status text,data text)');record={'rejected':[]};state={'deadline_utc':'2026-10-27T05:31:00Z'}
 m['restore_supervision'](p,{'id':'dead1234abcd'}, {},state,db,time.time()+120,record)
 queue=json.loads((d/'ROOT_CONTROL_QUEUE.json').read_text());assert len(queue['tasks'])==1 and queue['tasks'][0]['ready'];assert queue['tasks'][0]['role']=='coordinator';assert pathlib.Path(queue['tasks'][0]['directory'],'TASK.txt').exists();assert pathlib.Path(queue['tasks'][0]['directory'],'SCOPE.json').read_text().strip()=='[]'
 m['restore_supervision'](p,{'id':'dead1234abcd'}, {},state,db,time.time()+120,record);assert len(json.loads((d/'ROOT_CONTROL_QUEUE.json').read_text())['tasks'])==1
 group=json.loads((p/'GROUP.json').read_text());assert group['supervisor_self_job'].endswith('/SELF_JOB.json');assert group['previous_supervisor_self_job']==str(p/'SELF_JOB.json');print('Coordinator replacement CPU checks: unique/idempotent queue, zero pair scope, concrete task, explicit supervision pointer passed; no native launch')
