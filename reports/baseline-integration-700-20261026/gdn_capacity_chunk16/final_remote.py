import json,hashlib,datetime,subprocess
from pathlib import Path
W=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads((W/'ORIGINAL_EVIDENCE_PRESERVED.json').read_text())
preserved={c:{n:sha(Path(v['path']))==v['sha256'] for n,v in files.items()} for c,files in original.items()}
deployment=json.loads((W/'DEPLOYMENT_VERIFIED.json').read_text())
post={p:sha(Path(p))==v['expected'] for p,v in deployment.items()}
frozen=json.loads((W/'FROZEN_ADAPTER.json').read_text())
freeze={p:sha(Path(p))==v for p,v in frozen['files'].items()}
assert all(all(v.values()) for v in preserved.values()) and all(post.values()) and all(freeze.values())
rows=json.loads((W/'RESULT.json').read_text());eligible=[r['exactkey'] for r in rows if r['status']=='verified_pending_parent_review']
r=json.loads((W/'REGISTRATION_CANDIDATES.json').read_text());r.update(status='fullcheck_verified_pending_parent_serial_registration',exactkeys=eligible,registered=False,accepted_added=0);(W/'REGISTRATION_CANDIDATES.json').write_text(json.dumps(r,indent=2))
p=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True)
(W/'POST_GUARD_STATUS.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2))
(W/'POST_RUN_INTEGRITY.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_evidence_preserved':preserved,'deployment_pins_unchanged':post,'frozen_adapter_unchanged':freeze,'registered':False,'accepted_added':0},indent=2))
print(json.dumps({'eligible':len(eligible),'deployment_verified':len(post),'original_evidence_preserved':True,'frozen_adapter_unchanged':True,'guard_status_rc':p.returncode}))
