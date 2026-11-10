import collections,datetime,hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];scope=json.loads((P/'SCOPE.json').read_text());d=json.loads((P/'RESULT.json').read_text());rows={(r['baseline'],r['operator'],r['case_id']):r for r in d['rows']};events=[str(P/n/'guard.log') for n in ['batch_w4','batch_probes','batch_w4_retry']]
api={'GemmFwdOp':'MatMulV3 via original classicpilot_nn graph JIT','GroupedGemmFwdOp':'MatMulV3 per logical group via original classicpilot_nn','FusedAddRMSNormFwdOp':'AddRmsNorm with private input clones','FusedAddLayerNormFwdOp':'AddLayerNorm','BatchNormFwdOp':'BatchNormV3 with canonical input dtype','InstanceNormFwdOp':'LayerNormV4 per channel / BatchNormV3 evaluation; ops-math original LayerNorm example','GemmFP8FwdOp':'QuantBatchMatmulV3, group_size=4303356032','ProdFwdOp':'UnsortedSegmentProd full reduction','INT8DequantPerChannelFwdOp':'Mul with exact FP32 widening and row-scale broadcast'}
for s in scope:
 k=(s['baseline'],s['operator'],s['case_id'])
 if k not in rows:
  rows[k]=dict(baseline=k[0],operator=k[1],case_id=k[2],status='blocked',reason='Not device-executed: three separate guarded batches were interrupted by external UID1003 NPU processes (guard76). Original complete-API binding is prepared; stopped further repetitions after persistent host interference. This is an execution blocker, not an API eligibility exclusion.',api=api.get(k[1]),guard_receipt=None,checker_receipt=None,checked_outputs=0,total_outputs=2 if k[1].startswith('FusedAdd') else 1,registered=False,factory=None,changed_files=[str(P/'adapter.py'),str(P/'run_case.py')],timings=dict(queue_s=None,device_s=None,compile_s=None),evidence=events+[str(P/'SCOPE.json'),str(P/'adapter.py')],device_executed=False)
 if k[1] in ('GemmFwdOp','GroupedGemmFwdOp'):
  rows[k]['reason']='install3 MatMulV3 uses a forbidden internal registration patch. Found an existing original classicpilot_nn deployment with 144/144 source files matching original, and prepared a direct full MatMulV3 binding; its scheduled representative was not reached before repeated external UID1003 guard76 interruption. No numerical claim for this original deployment.'
  rows[k]['api']=api[k[1]];rows[k]['evidence']=rows[k].get('evidence',[])+[str(P/'CLASSIC_MATMUL_SOURCE_COMPARE.json')]+events
  rows[k]['device_executed']=False
 if k[1]=='BatchNormFwdOp':rows[k]['total_outputs']=2 if k[2]=='large-spatial-float16' else 1
 if k[1]=='GemmW4A16FwdOp' and rows[k]['status']!='completed':
  rows[k]['reason']='Original full WeightQuantBatchMatmulV2 route has an accepted independent representative. Both continuation batches were interrupted by external UID1003 contexts (guard76); any checker passes within interrupted batches are retained but not admitted. No further repetition after third host interference event.'
  rows[k]['evidence']=rows[k].get('evidence',[])+events
 for p in P.glob('*/checker.json'):
  c=json.loads(p.read_text())
  if (c['library'],c['op'],c['case_id'])==k:
   rows[k].setdefault('attempt_history',[]).append(dict(checker_receipt=str(p),checker_status=c.get('status'),verification=c.get('verification'),guard_receipt=str(p.parent/'GUARD_RESULT.json'),not_necessarily_accepted=True))
d.update(rows=[rows[(s['baseline'],s['operator'],s['case_id'])] for s in scope],in_progress=False,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=dict(collections.Counter(r['status'] for r in rows.values())),stop_reason='Repeated external UID1003 device use: 3 guard76 interruptions; no process termination/device reset/bypass.')
assert len(d['rows'])==66 and len(rows)==66
f=P/'RESULT.tmp';f.write_text(json.dumps(d,indent=2));f.replace(P/'RESULT.json')
h=dict(utc=d['finished_at'],phase='finished_with_host_blockers',current_keys=[],last_progress_at=d['finished_at'],blocked_reason=d['stop_reason'],summary=d['summary']);(P/'HEARTBEAT.json').write_text(json.dumps(h,indent=2));print(d['summary'])
