from pathlib import Path
import os,time,datetime,json
from orchestrate import run,beat,P
# One foreground device command at a time. Existing B64 submission owns the slot until its result is saved.
waitfor=P/'attempts/mamba2-2.7b-decode-b64-bfloat16/RESULT.json'
while not waitfor.exists():time.sleep(10)
order=['mla-tp-b64-float16','256M-bfloat16','mla-tp-b6-float16','ds-v2-4k-float16','mamba2-2.7b-decode-b256-bfloat16','qwen3-next-decode-b64-float16','mla-tp-b2-float16','mla-tp-b64-bfloat16','mla-tp-b6-bfloat16','ds-v2-4k-bfloat16']
limit=datetime.datetime(2026,10,27,5,31,tzinfo=datetime.timezone.utc).timestamp()
for i,cid in enumerate(order):
 if time.time()+735>limit:
  (P/'DEADLINE_REMAINING.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unsubmitted':order[i:],'reason':'Insufficient remaining deadline for wait120 plus run600 and collection'},indent=2));break
 os.environ['HOST_PREFERENCE']=['openlibing-DevEnv-661495','openlibing-DevEnv-055409'][i%2]
 run(cid)
(P/'BATCH_COMPLETE.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
