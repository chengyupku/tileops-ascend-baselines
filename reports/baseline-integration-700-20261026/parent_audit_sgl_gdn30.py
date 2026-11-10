from pathlib import Path
import json,hashlib,datetime,subprocess
W=Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';O=D/'sgl_gdn_remote38';R=D/'sgl_gdn_register30';RW=Path('/opt/tileops-all-pairs/workspace');RO=RW/O.relative_to(W)
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in read(R/'ARTIFACT_HASHES.json').items():assert sha(R/p)==h,p
rec=read(R/'RECOMMENDED_ACCEPT.json');assert len(rec)==30;mapping=read(R/'SOURCE_MAPPING.json');remote=dict(mapping['all_dependency_hashes']);remote.update(mapping['helper_hashes']);remote[mapping['binding']['registered_module']]=mapping['binding']['sha256'];out=[];loaded_checks=0
for row in rec:
 k=row['key'];cp=O/row['checker_receipt'];gp=O/row['guard_receipt'];assert sha(cp)==row['checker']['sha256'];assert sha(gp)==row['guard']['sha256'];c=read(cp);g=read(gp);v=c['verification'];assert c['status']=='verified' and v['checked_outputs']==v['total_outputs']==2 and not v['unchecked_reason'];assert c['inputs_preserved'] and not c['oracle_source_changed'] and not any(c.get(x) for x in ('error','sync_error','cleanup_error'));assert g.get('exit_code',g.get('returncode'))==0
 assert c['input_seed']==int.from_bytes(hashlib.sha256((k[1]+'::'+k[2]).encode()).digest()[:4],'big');assert c['verifier_sha256']==sha(W/'TileOPs/scripts/ascend_pair_check.py');assert c['case_snapshot']==row['canonical_snapshot']
 assert c['source_sha256'][row['tested_binding']['path']]==mapping['binding']['sha256']==sha(R/'frozen_verified/integration700_sgl_gdn_remote38.py');remote.update(c['source_sha256'])
 for p,h in c['oracle_source_sha256'].items():remote[str(RW/p) if not Path(p).is_absolute() else p]=h
 for label in ['checker','guard','loaded_modules_receipt','actual_route_receipt','maps_receipt','native_receipt']:
  x=row[label];local=W/Path(x['path']).relative_to(RW);assert sha(local)==x['sha256'];remote[x['path']]=x['sha256']
 lm=read(W/Path(row['loaded_modules_receipt']['path']).relative_to(RW));assert lm['sgl_kernel_npu.fla.chunk']['sha256']==mapping['api_source']['sha256'];assert lm['workloads.numerics']['sha256']==mapping['original_numerics'];loaded_checks+=1
 route=read(W/Path(row['actual_route_receipt']['path']).relative_to(RW));assert route['key']==k and route['provider']==row['tested_binding']['path'];assert row['loaded_sgl_library']['path'] in (W/Path(row['maps_receipt']['path']).relative_to(RW)).read_text()
 assert not c['case_snapshot']['parameters'].get('use_qk_l2norm_in_kernel',False)
 out.append({'key':k,'checker':str(cp),'guard':str(gp),'full_outputs':v,'guard_returncode':0})
# Relative source keys are workspace-relative. Validate exact frozen original module as well.
remote={str(RW/p) if not Path(p).is_absolute() else p:h for p,h in remote.items()}
remote.pop(str(RW/'TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py'),None)
remote.pop(str(RO/'integration700_sgl_gdn_remote38.py'),None)
remote[str(RW/R.relative_to(W)/'frozen_verified/integration700_sgl_gdn_remote38.py')]=mapping['binding']['sha256']
# Review-derived fresh-import CPU checker: stubs runtime calls and prohibits actual NPU imports; remove receipt write.
code=(R/'fresh_import_check.py').read_text();code=code.replace("(Q/'FRESH_IMPORT_CHECKS.json').write_text(json.dumps(checks,indent=2))",'pass')
pre='import pathlib,hashlib\nfor p,h in '+repr(remote)+'.items():\n assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p\n'
proc=subprocess.run([str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'),'openlibing-DevEnv-055409','python3 -S -'],input=pre+code,text=True,capture_output=True,timeout=50);assert proc.returncode==0,proc.stderr
live=json.loads(proc.stdout);live.update(remote_hashes_verified=len(remote),historical_loaded_original_api_and_numerics=loaded_checks,npu_runs=0,write_operations=0);(D/'PARENT_SGL_GDN30_LIVE_CHECK.json').write_text(json.dumps(live,indent=2))
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualified_rows':out,'remote_hashes_verified':len(remote),'canonical_full_outputs':60,'original_boundary':'Original upstream complete chunk_gated_delta_rule_native, unmodified default chunk size and internal float32 recurrence. Original function itself uses PyTorch/CANN, not a custom fused SGL device kernel. Adapter only zero Delta gate, independent sequence slicing/head replication/state orientation; exact bytes match tested historical source hashes.','proof_limit':mapping['runner_archive_note'],'runtime_proof':'Per-case loaded original chunk API and shared original numerics hashes, actual route, loaded-library maps, full canonical checker and clean guard. CANN launch logs include reference and candidate and are not claimed as custom SGL kernel attribution.','fresh_registry':live,'new_npu_runs':0}
(D/'PARENT_SGL_GDN30_REVIEW.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({'qualified':len(out),'remote_hashes':len(remote),'new_npu_runs':0}))
