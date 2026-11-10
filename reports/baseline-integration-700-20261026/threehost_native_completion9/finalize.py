from pathlib import Path
import json,hashlib,time,datetime,shutil
P=Path(__file__).resolve().parent;S=P/'server1'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n')
def stamp(x):return datetime.datetime.fromisoformat(x).timestamp()
rows=[]
for scoped in load(P/'SCOPE.json'):
 cid=scoped['case_id'];op=scoped['operator'];D=S/cid;R=D/'result';r=load(R/'representative.json');events=load(R/'NATIVE_CALLS.json');ex=load(R/'EXECUTION.json');g=load(D/'GUARD_FINAL.json');native=load(D/'ACTUAL_NATIVE.json');cap=load(R/'CAPTURE.json');snap=load(R/'guard_snapshot.json')
 mode='fp8' if op=='BmmFP8FwdOp' else 'bf16arm2';freeze=load(S/('FROZEN_'+mode+'.json'))
 old=load(P.parent/('fp8_native_reconcile5' if mode=='fp8' else 'bf16_moe_hadamard_native10')/cid/('RESULT.json' if mode=='fp8' else 'representative.json'))
 seed=old['seed'] if mode=='fp8' else old['input_seed'];oracle=old['canonical_existing_evidence']['oracle_source_sha256'] if mode=='fp8' else old['oracle_source_sha256']
 checks={'full_checker_verified':r['status']=='verified','guard_exit_zero':g['exit_code']==0,'all_outputs_checked':r['verification']['checked_outputs']==r['verification']['total_outputs'] and r['verification']['total_outputs']>0,'inputs_preserved':r['inputs_preserved'],'original_seed':r['input_seed']==seed,'original_parameters':r['case_snapshot']==old['case_snapshot'],'original_oracle_hashes':r['oracle_source_sha256']==oracle,'frozen_unchanged':ex['frozen_unchanged'],'provider_unchanged':not r['provider_source_changed'],'oracle_unchanged':not r['oracle_source_changed']}
 enters=[e for e in events if e['event']=='enter'];returns=[e for e in events if e['event']=='return']
 checks['actual_workspace_and_launch']={e['function'] for e in enters}=={'aclnnGroupedMatmulV5','aclnnGroupedMatmulV5GetWorkspaceSize'}
 checks['native_returns_zero']=len(enters)==len(returns) and all(e['return_code']==0 for e in returns)
 checks['actual_host_pinned']=all(e['map_inode']==e['stat_inode'] and e['dladdr_path']==e['path'] and freeze['sha256'].get(e['path'])==e['sha256'] for e in enters)
 kernel='GroupedMatmul_'+('0db0da5b4b38b65b8b4dd21b626476e5' if mode=='fp8' else 'e84aaec0eccf7b8304da51ea5f5a2076')
 lines=[l for f in native['driver_evidence'] for l in f['lines']];reg=[l for l in lines if 'Register Kernel succ' in l and kernel in l];launch=[l for l in lines if 'StreamLaunchKernelV2' in l and kernel in l]
 checks['original_binary_register_and_launch']=bool(reg and launch)
 outputs={n:d for n,d in cap.items() if '_output' in n};checks['full_outputs_saved']=bool(outputs) and all((R/(n+'.bin')).is_file() and sha(R/(n+'.bin'))==d['sha256'] for n,d in outputs.items())
 assert all(checks.values()),(cid,checks)
 timings=ex['timings'];detail={'guard_queue_s':stamp(snap['acquired_at'])-stamp(snap['requested_at']),'physical_idle_wait_s':stamp(snap['updated_at'])-stamp(snap['acquired_at']),'reference_s':sum(v for k,v in timings.items() if k.startswith('reference')),'candidate_s':sum(v for k,v in timings.items() if k.startswith('candidate')),'fullchecker_wall_s':r['wall_seconds'],'guard_total_wall_s':g['wall_seconds'],'cpu_preparation_s':None,'cpu_preparation_note':'Prepared outside lock; not separately timed. Guard total includes process initialization and instrumentation.'}
 result={'key':['ops-transformer',op,cid],'accept_recommended':True,'accepted_added':0,'status':'new_canonical_verified_complete_native_evidence','host':'openlibing-DevEnv-661495','sdk':'9.1 ARM','checks':checks,'verification':r['verification'],'input_seed':r['input_seed'],'fullchecker':str(R/'representative.json'),'guard':str(D/'GUARD_FINAL.json'),'actual_native':str(D/'ACTUAL_NATIVE.json'),'native_host':native['actual_host_apis'],'kernel':kernel,'kernel_register_lines':reg,'kernel_launch_count':len(launch),'timings':detail,'prior_receipt_preserved':True,'not_retroactive':True,'evidence_sha256':{str(f.relative_to(P)):sha(f) for f in [R/'representative.json',R/'NATIVE_CALLS.json',R/'EXECUTION.json',R/'CAPTURE.json',D/'ACTUAL_NATIVE.json',D/'GUARD_FINAL.json',D/'guard.log']}}
 out=P/cid;out.mkdir(exist_ok=True);dump(out/'RESULT.json',result);rows.append(result)
for n in ['FROZEN_fp8.json','FROZEN_bf16arm2.json']:shutil.copyfile(S/n,P/n)
dump(P/'RESULT.json',{'scope_count':9,'accepted_added':0,'accept_recommended_true':9,'results':rows})
dump(P/'DELIVERY.json',{'status':'complete_pending_parent_review','scope_count':9,'new_canonical_verified':9,'accept_recommended_true':9,'accepted_added':0,'registration_module':str(P/'registration.py'),'registration_target':'Tested original ARM server1 deployment; CPU-lazy exact contracts, fail closed on unavailable/changed frozen closure; parent publishes','contracts':str(P/'CONTRACTS.json'),'contract_checks':load(P/'CONTRACT_CHECKS.json'),'results':str(P/'RESULT.json'),'remaining_issues':['Parent review and shared registry publication outstanding; worker has not changed shared registry or accepted ledger.'],'limitations':['ARM and local clocks differ by approximately 93 seconds: compare only same-host timestamps.','Remote cgroup-v1 capacity uses visible raw headroom and hierarchical limit; ancestor usage is not visible.','CPU preparation was not separately timed.','Full raw process logs and isolated runtime caches are retained in the remote private report; local ACTUAL_NATIVE contains matched lines and raw log hashes.'],'no_repeated_successful_numerical_tests':True,'new_numerical_case_count':9,'failed_pre_candidate_attempts':2,'prior_receipts_untouched':True})
dump(P/'HEARTBEAT.json',{'at':time.time(),'stage':'complete_pending_parent_review','verified':9,'accepted_added':0})
print('validated 9/9')
