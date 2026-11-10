import json,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;R=json.loads((P/'RESULT.json').read_text());scope=json.loads((P/'SCOPE.json').read_text());done={(r['baseline'],r['operator'],r['case_id']) for r in R['rows']};guard=str(P/'mixed02/GUARD_RESULT.json');log=str(P/'mixed02/guard.log')
for x in scope:
 k=(x['baseline'],x['operator'],x['case_id'])
 if k in done:continue
 if x['operator'] in ['GemmFwdOp','GroupedGemmFwdOp']:
  reason='Original BF16 MatMulV3 representative hit 507015/fixpipe invalid GM address; exact FP32 widening replacement probe was interrupted by external UID1003 guard76 before a valid verdict. Local route stopped immediately. Server2 available ops-nn host deployment retains a source dtype patch and is not admissible. This key was not device-validated; no algorithmic failure inferred from representative.'
  evidence=[str(P/'gemmclassic_isolated/case.log'),str(P/'MATRIX_FP32_REPAIR.md'),log,str(P/'server2_hostbuild.txt')];api='ops-nn::MatMulV3'
 else:
  reason='Not device-executed: external UID1003 process interrupted mixed02 with guard76; local route stopped per instruction. Server2 installed admissible ops-nn graph package exposes pooling/matrix operators only, not required BatchNormV3. No kernel/compiler rebuild or patched replacement attempted.'
  evidence=[log,str(P/'server2_remaining_probe.txt')];api='ops-nn::BatchNormV3'
 R['rows'].append(dict(baseline=k[0],operator=k[1],case_id=k[2],status='blocked',reason=reason,api=api,guard_receipt=guard,checker_receipt=None,checked_outputs=0,total_outputs=None,registered=False,factory=None,changed_files=[],timings=dict(queue_s=None,case_s=None,guard_total_s=None,compile_s=None),evidence=evidence,attempt=None))
for r in R['rows']:
 if r.get('attempt') in ['remote_instance01','remote_dequant01']:r['host']='server2';r['reason']='Existing remote original deployment blocked before target verification: '+r['reason'];r['evidence'] += [str(P/'server2_math_source_checks.txt'),str(P/'server2_math_ready.txt')]
 else:r['host']='local'
 if r['operator']=='GemmFwdOp' and r['case_id']=='ds-v3-m1-qkv-a-bfloat16':r['evidence'] += [str(P/'gemmclassic_isolated/checker.json'),str(P/'MATRIX_FP32_REPAIR.md')]
R['remaining_keys']=[];R['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();R['in_progress']=False
assert len(R['rows'])==48
q=P/'RESULT.json.tmp';q.write_text(json.dumps(R,indent=2));q.replace(P/'RESULT.json')
