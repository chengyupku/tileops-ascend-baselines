from pathlib import Path
import json,sys,shutil
W=Path(__file__).resolve().parent;scope=json.loads((W/'SCOPE.json').read_text());b=json.loads((W/'BLOCKERS.json').read_text());contracts=json.loads((W/'ALL_CONTRACTS.json').read_text());cs={(c['op'],c['case_id']):c for c in contracts}
for x in scope:
 op=x['operator'];i=x['case_id'];p=cs[op,i]['params'];k=op+'/'+i
 if op=='DeltaNetFwdOp':b[k]={'reason':'Original FLA oracle requires missing SDK9.2 ascend_all_cann_install.info; current reused layouts still lack metadata. No unchanged failing oracle rerun or synthetic metadata/compiler change.','api':'aclnnChunkKdaFwd / aclnnChunkGatedDeltaRule','evidence':['ORACLE_CURRENT.json','prior_evidence/ORACLE_DEPLOYMENT_DIAGNOSIS.json']}
 elif op=='GDNFwdOp' and p['q'][1]=='float16':b[k]={'reason':'FP16 requires original full ChunkKdaFwd. Previous original package codegen timed out300; no complete loadable package exists in reused deployment; RecurrentKda/ChunkGatedDeltaRule are BF16-only. No unchanged compile retry.','api':'aclnnChunkKdaFwd','evidence':['prior_evidence/FP16_DEPLOYMENT_BLOCKER.json','prior_evidence/CHUNK_BUILD_GUARD.json','prior_evidence/chunk-build.stderr.log']}
(W/'BLOCKERS.json').write_text(json.dumps(b,indent=2))
print('blockers',len(b))
