import json,pathlib,datetime,hashlib,shutil
P=pathlib.Path(__file__).resolve().parent;old=P.parent/'tla_attention';E=P/'original_remote_evidence';now=datetime.datetime.now(datetime.timezone.utc)
def save(n,d):
 t=P/(n+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(P/n)
for n in ['adapter.py','adapter.pto-v2.py','run_case.py','launch.py','API_REVIEW.json','SOURCE_HASHES.json','COMPILER_SOURCE_HASHES.json','DELIVERY.json','RESULT.json']:
 d=P/'original_metadata';d.mkdir(exist_ok=True);shutil.copy2(old/n,d/n)
runs=[]
for d in sorted((E/'runs').iterdir()):
 if not (d/'guard.json').exists():continue
 g=json.loads((d/'guard.json').read_text())
 if g['returncode']!=124:continue
 meta=json.loads((d/'GUARD_RUNNING.json').read_text());start=datetime.datetime.fromisoformat(meta['updated_at']).timestamp()
 src=list((d/'TMPDIR').glob('*.cpp'));so=list((d/'TMPDIR').glob('*.so'))
 logs=list(d.glob('ASCEND_PROCESS_LOG_PATH/run/plog/*'));ev=[]
 for f in logs:
  for i,l in enumerate(f.read_text(errors='replace').splitlines(),1):
   if 'three minutes timeout' in l:ev.append({'file':str(f.relative_to(P)),'line':i,'text':l})
 stack=list(d.glob('ASCEND_WORK_PATH/**/*.txt'))
 trace=list(d.glob('jit/launch-evidence/*/*.kernel.txt'))
 runs.append(dict(run=d.name,key=g['key'],phase='device_task_completion_wait',evidence=ev,stackcore=[str(f.relative_to(P)) for f in stack],jit_entered_evidence=[str(f.relative_to(P)) for f in trace],compiled_binaries=[dict(file=str(f.relative_to(P)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),mtime_utc=datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat()) for f in so],timings=dict(queue_seconds=datetime.datetime.fromisoformat(meta['acquired_at']).timestamp()-datetime.datetime.fromisoformat(meta['requested_at']).timestamp(),guarded_command_seconds=g['finished']-start,compiled_artifact_available_after_command_start_seconds=max([f.stat().st_mtime-start for f in so],default=None),cpu_compile_seconds=None,oracle_seconds=None,device_active_seconds=None),guard_receipt=str((d/'guard.json').relative_to(P))))
diag=dict(created_at=now.isoformat(),conclusion='All four repaired representatives reached compiled-device execution; runtime stream completion remained pending for three minutes. Not a cold-compile or cache-lock timeout. No evidence supports a slow FP64 oracle as the terminal stage.',confidence='Direct runtime plog evidence for all four; adapter return-before-synchronize evidence for three; MLA generated-source evidence proves entry into JIT invocation after oracle.',root_cause_limit='Device kernel/stream noncompletion is established; exact internal synchronization/compiler defect is not proven by these logs. Do not claim a particular kernel bug or mutate its schedule.',runs=runs,decision='No unchanged retry; no 900-second extension; no CPU precompile needed because original binaries already exist.',precompile='Complete literal compiler argv was not saved in old receipts. Existing completed binaries make reconstruction unnecessary.',constraints='No vendor/compiler/kernel/registry modifications; no oracle, case-size, precision or tolerance changes.',new_device_runs=0)
save('DIAGNOSIS.json',diag)
oldrows=json.loads((old/'RESULT.json').read_text())['rows'];lookup={(r['baseline'],r['operator'],r['case_id']):r for r in oldrows};rows=[]
for s in json.loads((P/'SCOPE.json').read_text()):
 key=tuple(s[k] for k in ['baseline','operator','case_id']);r=lookup[key];rep=r.get('representative_evidence')
 if not rep:
  rep=next((x.get('representative') for x in json.loads((old/'DELIVERY.json').read_text()).get('unresolved',[]) if tuple(x.get('key',[]))==key),None)
 if not rep:
  op=s['operator'];rep='mla-tp-b1-float16-pto-v2' if op=='MLADecodeWithKVCacheFwdOp' else 'serving-kv512-float16-pto-v2' if op=='MHADecodePagedWithKVCacheFwdOp' else 'gemma-2b-8k-bs1-float16-pto-v3' if 'gemma' in s['case_id'] else 'qwen3-30b-1k-float16-pto-v3'
 rr=next((x for x in runs if x['run']==rep),runs[0])
 rows.append(dict(zip(['baseline','operator','case_id'],key),status='blocked',reason='Shared original API route has compiled binary but runtime stream reports three-minute pending device tasks; no evidenced legal top-level repair. Sibling is not individually executed and is not declared mathematically unsupported.',api=r.get('api'),guard_receipt=rr['guard_receipt'],checker_receipt=None,checked_outputs=0,total_outputs=r.get('total_outputs'),registered=False,factory=None,changed_files=[],timings=dict(queue_seconds=0,device_seconds=0,compile_seconds=None),representative=rep,evidence=['DIAGNOSIS.json',*rr['evidence']],prior_row=r))
save('RESULT.json',dict(created_at=now.isoformat(),counts={'blocked':len(rows),'completed':0,'verified_unregistered':0},rows=rows))
start=datetime.datetime.fromisoformat(json.loads((P/'STARTED.json').read_text())['utc'])
save('DELIVERY.json',dict(deployment_host='server1 / openlibing-DevEnv-661495',remote_private='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/tla_attention_recovery01',finished_at=now.isoformat(),elapsed_seconds=(now-start).total_seconds(),counts={'blocked':len(rows),'completed':0,'verified_unregistered':0},new_runs=0,repeated_runs=0,queue_seconds=0,held_seconds=0,diagnosis='DIAGNOSIS.json',formal_entry_mappings=[{k:r[k] for k in ['baseline','operator','case_id','api','registered','factory']} for r in rows],source_hash='original_metadata/SOURCE_HASHES.json',binary_hashes='DIAGNOSIS.json',patch=None,patch_reason='No numerical success; no registration patch or deployable frozen module issued.',backup='original_metadata/',raw_evidence='original_remote_evidence.tar.gz',run_recipe='Historical exact commands and environments: original_metadata/DELIVERY.json; do not replay unchanged. Recovery reproduction: CPU-only python3 finalize_diagnosis.py using preserved evidence. Any future repaired device run must use owner baseline700-20261026/tla_attention_recovery01 with wait3600 timeout300.',unresolved='73 keys depend on routes with device stream noncompletion; exact internal root cause not proven, prohibited compiler/kernel repair not attempted.',checks=['73 unique scope keys matched prior result','Four distinct timeout routes have compiled shared objects and three-minute pending-task runtime evidence','No NPU imports or device runs this recovery','No shared or historical files changed'],changed_files=['STARTED.json','HEARTBEAT.json','DIAGNOSIS.json','RESULT.json','DELIVERY.json','finalize_diagnosis.py','original_metadata/','original_remote_evidence/','original_remote_evidence.tar.gz']))
save('HEARTBEAT.json',dict(utc=now.isoformat(),phase='delivered_blocked',current_keys=[],last_progress_at=now.isoformat(),blocked_reason=diag['conclusion']))
assert len(rows)==73 and len(set((r['baseline'],r['operator'],r['case_id']) for r in rows))==73
assert len(runs)==4 and all(r['evidence'] and r['compiled_binaries'] for r in runs)
print(json.dumps({'rows':len(rows),'diagnosed_routes':len(runs),'elapsed_seconds':(now-start).total_seconds()}))
