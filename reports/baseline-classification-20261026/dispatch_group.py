"""Finite native worker dispatch, no custom worker runtime or quota requests."""
import pathlib,subprocess,json,sys,time,os
d=pathlib.Path(sys.argv[1]);items=json.load(open(d/'BATCHES.json'));p=d/'GROUP_JOBS.json'
if p.exists():raise SystemExit('Already dispatched: inspect existing jobs, do not replay')
cli=['python3','/root/dyq/workspace/.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py','--native-script','/root/dyq/tools/codex-worker.py']
out={'parent':os.environ.get('CODEX_THREAD_ID'),'started_at':time.time(),'jobs':[]}
for t in items:
 a=cli+['spawn','--cwd','/root/dyq/workspace','--task-file',t['task_file'],'--write',t['directory'],'--model','gpt-6-astra','--effort','medium','--timeout','1200','--notify','queue','--sandbox','danger-full-access']
 start=time.time();r=subprocess.run(a,text=True,capture_output=True,timeout=45)
 v={'batch':t['batch'],'directory':t['directory'],'count':t['count'],'submitted_at':start,'returned_at':time.time(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 try:v['native']=json.loads(r.stdout)
 except ValueError:pass
 out['jobs'].append(v);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2));tmp.replace(p)
 print(json.dumps({'batch':t['batch'],'rc':r.returncode,'native':v.get('native'),'stderr':r.stderr},ensure_ascii=False),flush=True)
 if r.returncode:break
