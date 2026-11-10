import json,pathlib,subprocess,time,os
D=pathlib.Path(__file__).resolve().parent;plan=json.load(open(D/'PLAN.json'));out=D/'JOBS.json'
if out.exists():raise SystemExit('already submitted; inspect JOBS, no replay')
report={'parent':os.environ.get('CODEX_THREAD_ID'),'jobs':[]}
cli=['python3','/root/dyq/workspace/.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py','--native-script','/root/dyq/tools/codex-worker.py']
for w in plan['workers']:
 args=cli+['spawn','--cwd','/root/dyq/workspace','--task-file',w['task_file'],'--write',w['directory'],'--model','gpt-6-astra','--effort','medium','--timeout','10800','--notify','queue','--sandbox','danger-full-access','--npu-owner',w['owner'],'--npu-wait','3600','--npu-timeout','300']
 t=time.time();p=subprocess.run(args,text=True,capture_output=True,timeout=60)
 item=dict(w,submitted_at=t,returned_at=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 try:item['native']=json.loads(p.stdout)
 except ValueError:pass
 report['jobs'].append(item);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(out)
 print(json.dumps({'name':w['name'],'count':w['count'],'returncode':p.returncode,'native':item.get('native'),'stderr':p.stderr},ensure_ascii=False),flush=True)
 if p.returncode:break
