"""Read-only public workflow status. Never equate a between-job idle gap with completion."""
import urllib.request,json,pathlib,datetime,time
P=pathlib.Path(__file__).resolve().parent
url='https://api.github.com/repos/yyttt6/TileOPs.github.io/actions/workflows/ascend950-nightly.yml/runs?per_page=20'
now=datetime.datetime.now(datetime.timezone.utc)
try:
 for retry in range(3):
  try:
   d=json.load(urllib.request.urlopen(url,timeout=10));break
  except Exception:
   if retry==2:raise
   time.sleep(1)
 rows=[{k:r.get(k) for k in ['id','status','conclusion','event','created_at','updated_at','html_url','head_sha']} for r in d['workflow_runs']]
 active=[r for r in rows if r['status']!='completed']
 scheduled=[r for r in rows if r['created_at']>='2026-11-09T17:55:00Z']
 result={'checked_at':now.isoformat(),'source':url,'active':active,'tonight_runs':scheduled,'all_recent':rows,'nightly_completion_confirmed':bool(scheduled) and not active and all(r['status']=='completed' for r in scheduled),'rule':'Completion plus current remote guard idle/driver/FD/health checks required. If no run appeared, do not claim it finished; review schedule and live runner before deciding.'}
except Exception as e:result={'checked_at':now.isoformat(),'error':str(e),'nightly_completion_confirmed':False}
(P/'NIGHTLY_STATUS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='all_recent'},indent=2))
