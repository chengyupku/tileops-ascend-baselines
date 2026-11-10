import subprocess,json,time,pathlib,os
R=pathlib.Path(__file__).resolve().parent
if (R/'JOBS.json').exists():raise SystemExit('Already dispatched; inspect existing jobs')
plan=json.load(open(R/'PLAN.json'));out={'parent':os.environ.get('CODEX_THREAD_ID'),'jobs':[]}
cli=['python3','/root/dyq/workspace/.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py','--native-script','/root/dyq/tools/codex-worker.py']
for g in plan['groups']:
 start=time.time();p=subprocess.run(cli+['spawn','--cwd','/root/dyq/workspace','--task-file',g['task_file'],'--write',g['directory'],'--model','gpt-6-astra','--effort','medium','--timeout','2400','--notify','queue','--sandbox','danger-full-access'],text=True,capture_output=True,timeout=40)
 j=dict(g,submitted_at=start,returned_at=time.time(),returncode=p.returncode,stderr=p.stderr)
 try:j['native']=json.loads(p.stdout)
 except ValueError:j['stdout']=p.stdout
 out['jobs'].append(j);(R/'JOBS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({'name':g['name'],'count':g['count'],'rc':p.returncode,'id':j.get('native',{}).get('id'),'home':j.get('native',{}).get('home'),'stderr':p.stderr},ensure_ascii=False),flush=True)
 if p.returncode:break
