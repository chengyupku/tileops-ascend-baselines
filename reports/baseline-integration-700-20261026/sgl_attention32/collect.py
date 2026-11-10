from pathlib import Path
import json,datetime,sys
R=Path(__file__).resolve().parent
attempt=sys.argv[1];a=R/'runs'/attempt
cmd=json.loads((a/'COMMAND.json').read_text());op,cid=cmd[-3:-1]
g=json.loads((a/'GUARD_RESULT.json').read_text());cpath=a/'output/checker.json';c=json.loads(cpath.read_text()) if cpath.exists() else {}
v=c.get('verification',{});passed=c.get('status')=='verified' and v.get('checked_outputs',0)==v.get('total_outputs',-1) and g['returncode']==0
running=a/'output/GUARD_RUNNING.json';m=json.loads(running.read_text()) if running.exists() else {}
dt=lambda x:datetime.datetime.fromisoformat(x)
queue=(dt(m['updated_at'])-dt(g['started'])).total_seconds() if m.get('updated_at') else None
row=dict(baseline='sgl-kernel-npu',operator=op,case_id=cid,status='verified_unregistered' if passed else 'failed',reason='Complete original checker passed; private-only write authorization prevents shared registration.' if passed else c.get('reason',c.get('error',f"guard exit {g['returncode']}")),api=c.get('api'),guard_receipt=str(a/'GUARD_RESULT.json'),checker_receipt=str(cpath) if cpath.exists() else None,checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs',1),registered=False,factory=json.loads((a/'output/ROUTE.json').read_text()).get('factory') if (a/'output/ROUTE.json').exists() else None,changed_files=[],timings=dict(queue_seconds=queue,device_seconds=g['elapsed_seconds']-queue if queue is not None else None,compile_seconds=None,total_seconds=g['elapsed_seconds']),source_sha256=c.get('source_sha256'),attempt=attempt)
d=json.loads((R/'RESULT.json').read_text());d['rows']=[r for r in d['rows'] if (r['operator'],r['case_id'])!=(op,cid)]+[row];d['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(R/'RESULT.json.tmp').write_text(json.dumps(d,indent=2));(R/'RESULT.json.tmp').replace(R/'RESULT.json');print(json.dumps(row,indent=2)[:4000])
