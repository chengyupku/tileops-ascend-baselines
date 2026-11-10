from pathlib import Path
import json,hashlib,datetime
T=Path(__file__).resolve().parent
scope=json.loads((T/'SCOPE.json').read_text());rows=[]
for s in scope:
 k=s['case_id'];p=T/k;v=p/'VALIDATION.json';r=p/'representative.json';g=p/'GUARD_RECEIPT.json'
 guard=json.loads(g.read_text()) if g.exists() else None
 receipt=json.loads(r.read_text()) if r.exists() else None
 if v.exists():status='verified_unregistered';reason='Same-host original full canonical passed; complete original native V5 FP32 API, BF16 lossless input widening/output cast, local x86 host and Ascend950 device build. Frozen pins, full outputs, preserved inputs and native loader/launch verified.'
 elif guard and guard['returncode']==124:status='blocked';reason='300-second whole canonical guard timeout. Incomplete execution is not a numerical failure; inspect captures for last completed stage. No unchanged retry.'
 elif receipt and receipt.get('status')=='failed':status='failed';reason=receipt.get('error','Canonical checker reported failure')
 elif guard:status='blocked';reason='Execution incomplete or resource/environment/guard rejection; no complete numerical result. See guard logs.'
 else:status='blocked';reason='Not executed; batch stopped before this exact key. No numerical failure established.'
 caps=json.loads((p/'CAPTURE.json').read_text()) if (p/'CAPTURE.json').exists() else {}
 row=dict(baseline=s['baseline'],operator=s['operator'],case_id=k,status=status,reason=reason,registered=False,accepted_added=0,factory='binding:dense_fp32',environment={'ASCEND_CUSTOM_OPP_PATH':str(T/'install/vendors/bf16gmm_transformer'),'ASCEND_CLASSIC_TRANSFORMER_VENDOR':str(T/'install/vendors/bf16gmm_transformer')},guard_receipt=str(g.relative_to(T)) if guard else None,checker_receipt=str(r.relative_to(T)) if receipt else None,verification=receipt.get('verification') if receipt else None,guard_returncode=guard['returncode'] if guard else None,wall_seconds=guard['elapsed_seconds'] if guard else None,capture_stages=list(caps),evidence=[str(x.relative_to(T)) for x in p.glob('*.json')] if p.exists() else [])
 row['original_reference_device']=caps.get('reference_0_output',{}).get('original_device')
 row['original_reference_dtype']=caps.get('reference_0_output',{}).get('dtype')
 rows.append(row)
summary=dict(worker='f671baf22ab041bf',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope_count=18,verified_unregistered=sum(x['status']=='verified_unregistered' for x in rows),blocked=sum(x['status']=='blocked' for x in rows),failed=sum(x['status']=='failed' for x in rows),accepted_added=0,registered=0,canonical_runs=sum(x['guard_receipt'] is not None for x in rows),rows=rows)
(T/'RESULT.json').write_text(json.dumps(summary,indent=2));(T/'RESULT.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
builds=[]
for name in ['build1','package','package-enabled','device-compile']:
 p=T/name/'GUARD_RECEIPT.json'
 if p.exists():builds.append(dict(stage=name,**json.loads(p.read_text())))
(T/'BUILD_TIMINGS.json').write_text(json.dumps({'stages':builds,'total_wall_including_queue_s':sum(x['elapsed_seconds'] for x in builds),'limit_build_hold_s':1800,'package_correction':'First --jit built host without ENABLE_BUILD_PKG; package target absent (rc2). Corrected command to original --pkg --jit, successfully packaged. No original compiler/kernel/source options changed.'},indent=2))
# Attest all actually mapped shared runtime libraries, including host dependencies.
mapped=set()
for p in T.glob('*/EXECUTION.json'):
 mapped.update(json.loads(p.read_text()).get('loaded_files',[]))
mp={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sorted(mapped) if Path(s).is_file()}
(T/'MAPPED_RUNTIME_FILEHASHES.json').write_text(json.dumps(mp,indent=2))
source=json.loads((T/'SOURCE_PINS.json').read_text());changed=[s for s,h in source.items() if hashlib.sha256(Path(s).read_bytes()).hexdigest()!=h]
deploy=json.loads((T/'DEPLOYMENT_MANIFEST.json').read_text());dc=[s for s,h in deploy.items() if hashlib.sha256((T/s).read_bytes()).hexdigest()!=h]
extra=sorted(str(p.relative_to(T)) for p in (T/'install/vendors/bf16gmm_transformer').rglob('*') if p.is_file() and str(p.relative_to(T)) not in deploy)
assert not extra,extra
(T/'POST_INTEGRITY.json').write_text(json.dumps({'source_files':len(source),'source_changed':changed,'deployment_files':len(deploy),'deployment_changed':dc,'accepted_added':0},indent=2));assert not changed and not dc
summary.update(boundary='ORIGINAL_BOUNDARY.json',first_case_evidence='FIRST_CASE_CAUSAL_EVIDENCE.json',sourcepins='SOURCE_PINS.json',dependency_pins='DEPENDENCY_FILEHASHES.json',runtime_pins='RUNTIME_FILEHASHES.json',actual_mapped_runtime_pins='MAPPED_RUNTIME_FILEHASHES.json',deployment_manifest='DEPLOYMENT_MANIFEST.json',post_integrity='POST_INTEGRITY.json',remaining_issues=[x['case_id']+': '+x['reason'] for x in rows if x['status']!='verified_unregistered'],parent_registration_required=True)
(T/'DELIVERY.json').write_text(json.dumps(summary,indent=2))
(T/'HEARTBEAT.json').write_text(json.dumps({'at':summary['utc'],'stage':'completed_delivery','verified_unregistered':summary['verified_unregistered'],'blocked':summary['blocked'],'failed':summary['failed'],'accepted_added':0}))
h={str(p.relative_to(T)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(T.rglob('*')) if p.is_file() and p.name!='FILEHASHES.json'}
(T/'FILEHASHES.json').write_text(json.dumps(h,indent=2));print(json.dumps({k:v for k,v in summary.items() if k in ['scope_count','verified_unregistered','blocked','failed','accepted_added','canonical_runs','remaining_issues']}))
