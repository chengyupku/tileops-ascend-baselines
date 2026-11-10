from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parent;i='q35-397b-tp2-65536-bfloat16';b=json.loads((W/'BLOCKERS.json').read_text());g=W/i/'GUARD_RECEIPT.json'
b['GDNFwdOp/'+i]={'reason':'Capacity fell during guard queue. Inside acquired reservation, original admission found effective54538618880 < required54886662144 bytes; checker/candidate not invoked. No unchanged numeric retry.','api':'aclnnChunkGatedDeltaRule','evidence':[i+'/fresh_resource_admission.json','batch_bounded_5/guard.stderr.log'],'guard_receipt':{'path':str(g.relative_to(W)),'sha256':hashlib.sha256(g.read_bytes()).hexdigest()},'checker_receipt':None}
(W/'BLOCKERS.json').write_text(json.dumps(b,indent=2))
