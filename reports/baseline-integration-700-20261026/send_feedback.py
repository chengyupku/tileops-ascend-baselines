import json,pathlib,subprocess,sys,datetime,sqlite3
D=pathlib.Path(__file__).resolve().parent
names=sys.argv[1].split(',');msg=sys.argv[2]
con=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
for j in json.load(open(D/'JOBS.json'))['jobs']:
 if names!=['all'] and j['name'] not in names:continue
 status=con.execute('select status from jobs where id=?',(j['native']['id'],)).fetchone()[0]
 if status not in ('running','starting'):
  print(j['name'],'skipped terminal',status);continue
 wd=pathlib.Path(j['directory']);f=wd/'FEEDBACK.txt'
 with f.open('a') as s:s.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+msg+'\n')
 events=pathlib.Path(j['native']['directory'])/'events.jsonl';tid=next(json.loads(x)['thread_id'] for x in events.read_text().splitlines() if json.loads(x).get('type')=='thread.started')
 p=subprocess.run(['/root/dyq/agent-bin/'+j['native']['home']['name'],'queue','--thread',tid,'--message',msg],capture_output=True,text=True,timeout=30)
 print(j['name'],p.returncode,p.stdout.strip(),p.stderr.strip())
