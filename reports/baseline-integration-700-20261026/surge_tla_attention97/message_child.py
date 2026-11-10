import pathlib,json,subprocess,os,sys,datetime
P=pathlib.Path(__file__).resolve().parent
r=next(r for r in json.loads((P/'CHILD_JOBS.json').read_text())['jobs'] if pathlib.Path(r['directory']).name==sys.argv[1]);n=r['native'];tid=None
for l in (pathlib.Path(n['directory'])/'events.jsonl').open():
 e=json.loads(l)
 if e.get('type')=='thread.started':tid=e['thread_id'];break
env=os.environ.copy();env['CODEX_HOME']=n['home']['path'];z=subprocess.run(['/root/dyq/agent-bin/codex','queue','--thread',tid,'--message',sys.argv[2]],env=env,capture_output=True,text=True,timeout=20)
with (P/'MESSAGES.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker':sys.argv[1],'message':sys.argv[2],'code':z.returncode,'stdout':z.stdout,'stderr':z.stderr},ensure_ascii=False)+'\n')
print(z.returncode,z.stdout,z.stderr)
