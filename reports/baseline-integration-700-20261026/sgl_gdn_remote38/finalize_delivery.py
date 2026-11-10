from pathlib import Path
import json,hashlib,datetime,collections,subprocess
P=Path(__file__).resolve().parent
result=json.loads((P/'RESULT.json').read_text());rows=result['rows']
for r in rows:
 if r['status']=='pending' and r['operator']=='GDNFwdOp':r.update(status='blocked',reason='Not run after representative prefill failed in unchanged FP64 gdn_ref MatMul (DT_DOUBLE rejected); no oracle change authorized.',shared_failure_receipt='runs/batch06--GDNFwdOp--prefill-narrow-state-bfloat16/checker.json',guard_receipt='batch06/GUARD_RESULT.json')
 # Any remaining Delta path needs the actual final representative verdict supplied before finalization.
 if r['status']=='pending' and (P/'DELTA_BLOCKER.json').exists():
  blocker=json.loads((P/'DELTA_BLOCKER.json').read_text());r.update(status='blocked',reason=blocker['reason'],shared_failure_receipt=blocker['checker_receipt'],guard_receipt=blocker['guard_receipt'])
now=datetime.datetime.now(datetime.timezone.utc);result['utc']=now.isoformat();(P/'RESULT.json').write_text(json.dumps(result,indent=2))
subprocess.run(['python3',str(P/'make_registration.py')],check=True)
success=[r for r in rows if r['status']=='verified_unregistered'];audit=[]
for r in success:
 f=P/r['checker_receipt'];c=json.loads(f.read_text());m=json.loads((f.parent/'output/LOADED_MODULES.json').read_text());g=json.loads((P/r['guard_receipt']).read_text());assert c['status']=='verified' and g['exit_code']==0
 assert c['verification']['checked_outputs']==c['verification']['total_outputs']==2
 assert not c.get('provider_source_changed') and not c.get('oracle_source_changed')
 assert m['sgl_kernel_npu.fla.chunk']['sha256']=='fdb3cd8654e664b4b2734c504c7bf5a2cb1e975596cd97a7762a8ac95936205c'
 assert m['sgl_kernel_npu.fla.chunk']['path']=='/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py'
 assert m['workloads.numerics']['path']=='/opt/tileops-all-pairs/workspace/TileOPs/workloads/numerics.py'
 assert m['workloads.numerics']['sha256']==hashlib.sha256((P/'numerics.original.py').read_bytes()).hexdigest()
 adapter='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_gdn_remote38/integration700_sgl_gdn_remote38.py'
 assert c['source_sha256'][adapter]==hashlib.sha256((P/'integration700_sgl_gdn_remote38.py').read_bytes()).hexdigest()
 audit.append({'key':r['key'],'checker_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'original_numerics':True,'all_outputs':True,'clean_guard':True,'adapter_bytes_match':True})
(P/'FINAL_CHECKS.json').write_text(json.dumps({'rows':audit,'scope':len(rows),'success_count':len(success),'no_shared_registration':True},indent=2))
guards={str(f.relative_to(P)):json.loads(f.read_text()) for f in P.glob('*/GUARD_RESULT.json')};start=datetime.datetime.fromisoformat(json.loads((P/'STARTED.json').read_text())['time']);elapsed=(now-start).total_seconds()
summary={'utc':now.isoformat(),'host':'server2','scope_count':len(rows),'counts':dict(collections.Counter(r['status'] for r in rows)),'verified_unregistered':len(success),'registered':0,'actual_work_seconds':elapsed,'verified_per_hour':len(success)*3600/elapsed,'guard_requests':len(guards),'timings':{'guard_total_s':sum(g['total_s'] for g in guards.values()),'queue_plus_idle_s':sum(g.get('queue_s') or 0 for g in guards.values()),'device_process_s':sum(g.get('device_s') or 0 for g in guards.values()),'compile_s':None,'kernel_latency_s':None},'guard_receipts':guards,'recommended_accept':'RECOMMENDED_ACCEPT.json','registration_patch':'REGISTRATION.patch','audit':'FINAL_CHECKS.json','native_evidence':'evidence/NATIVE_EVIDENCE.json','source_semantics':'Unmodified upstream complete chunk_gated_delta_rule_native API, internally torch/CANN; no claim of SGL custom fused kernel. All success uses original shared numerics. Probe01 alone had authorized diagnostic-only override and failed missing oracle dependency.','remaining':[{'key':r['key'],'reason':r['reason']} for r in rows if r['status']!='verified_unregistered']}
(P/'DELIVERY.json').write_text(json.dumps(summary,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps({'utc':now.isoformat(),'stage':'finished','counts':summary['counts']},indent=2));print(summary['counts'])
