"""CPU-only evidence reconciliation; never infer support from a status alone."""
import json,hashlib,time,shutil
from pathlib import Path
P=Path(__file__).resolve().parent
D=json.loads((P/'DECISIONS.json').read_text())
rows=json.loads((P/'HISTORICAL_ALL_VERIFIED_CANDIDATES.json').read_text())
audit={x['id']:x for x in json.loads((P/'HISTORICAL_NATIVE_SOURCE_AUDIT.json').read_text())}
accepted=[];rejected=[]
for row in rows:
 i=row['id']
 if i in D:continue
 c=json.loads(Path(row['path']).read_text());a=audit[i];lib,op,case=row['key']
 # Reviewed original full public APIs. Retired modified MatMul registration
 # and reference-only examples are deliberately not admitted here.
 if lib=='ops-nn' and not op.startswith(('VarFwd','VarMean')):continue
 if lib=='tilelang' and op!='GemmINT8W8A8FwdOp':continue
 if lib not in ('ops-nn','tilelang','tilelang-ascend','vllm-ascend','flash-linear-attention-npu','ops-transformer'):continue
 if not a['native_matched'] or a['native_unmatched']:continue
 oracle=c.get('oracle_source_sha256',{});checks={}
 for n,h in oracle.items():
  f=Path(n)
  if not f.exists() and '/workspace/' in n:f=Path(n.split('/workspace/',1)[1])
  checks[n]=f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==h
 v=c.get('verification',{})
 if not(row['params_match'] and oracle and all(checks.values()) and c.get('status')=='verified' and c.get('inputs_preserved') is True and c.get('provider_source_changed') is False and c.get('oracle_source_changed') is False and v.get('checked_outputs',0)>0 and v.get('checked_outputs')==v.get('total_outputs') and not v.get('unchecked_reason')):
  rejected.append(i);continue
 D[i]={'id':i,'key':row['key'],'bucket':'baseline支持','reason_kind':'historical_full_checker_verified_exact_contract','reason':'Original full checker verified all declared outputs with preserved inputs and identical exact parameters. Current oracle/checker hashes and recorded original native sources match. Reviewed bounded adaptation uses original complete APIs, with no changes to original kernel/compiler/schedule/checker; recent admission/deployment failures do not negate this historical correct execution. No performance-cache acceptance or fresh execution is claimed.','evidence':{'checker':row['path'],'checker_sha256':hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),'api':c.get('api'),'adaptation':c.get('adaptation'),'verification':v,'current_oracle_matches':checks,'native_source_audit':a,'composition_authority':'.worktrees/TileOPs-baseline-catalog/docs/ascend950/baselines/route_retirements.json','cache_recovered':False},'at':time.time()};accepted.append(i)
if accepted:
 f=P/'DECISIONS.json';shutil.copy2(f,P/f'DECISIONS.before-historical-reviewed-{int(time.time())}.json');tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(D,ensure_ascii=False,indent=2));tmp.replace(f)
(P/'HISTORICAL_RECONCILIATION.json').write_text(json.dumps({'at':time.time(),'accepted':accepted,'not_admitted':rejected},indent=2));print('accepted',len(accepted),'not_admitted',len(rejected))
