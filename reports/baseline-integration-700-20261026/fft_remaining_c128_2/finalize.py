from pathlib import Path
import json,hashlib,datetime,re,ast
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[]
for case in ['throughput-4k-complex128','scientific-1m-complex128']:
 d=P/case
 if not (d/'GUARD_RESULT.json').exists():continue
 c=read(d/'checker.json');g=read(d/'GUARD_RESULT.json');ch=read(d/'CHILD_RESULT.json');payload=read(d/'PAYLOADS.json');cleanup=read(d/'CLEANUP.json')
 assert c['status']=='verified' and g['returncode']==ch['returncode']==0
 assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 assert cleanup['synchronized_and_cleanup'] and payload[0]['input_preserved']
 logs=sorted((d/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'));events=[];indexes=[]
 for f in logs:
  lines=f.read_text(errors='replace').splitlines();selected=set()
  for i,line in enumerate(lines):
   if re.search(r'STFT|Stft|libcpu_kernels\.so|RunCpuKernel|aicpu_kernel.json',line):selected.update(range(max(0,i-2),min(len(lines),i+3)))
  for i in sorted(selected):events.append({'file':str(f),'line':i+1,'text':lines[i]})
  indexes.append({'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size,'selected_lines':[i+1 for i in sorted(selected)]})
 (d/'native.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events));write(d/'NATIVE_EVIDENCE_INDEX.json',indexes)
 assert any('RunCpuKernel[STFT] success' in e['text'] for e in events)
 assert any('libcpu_kernels.so success' in e['text'] for e in events)
 devpath=Path('/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libcpu_kernels.so')
 write(d/'DEVICE_BINARY_IDENTITY.json',{'logged_device_path':str(devpath),'host_readable':devpath.is_file(),'sha256':sha(devpath) if devpath.is_file() else None,'ownership':'system CANN AiCPU STFT via original complete ops-math aclStft; not custom libmath_aicpu_kernels.so','limit':'No SDK scan or device download; inaccessible device binary hash not claimed.'})
 files=[]
 for f in sorted(d.glob('*.pt')):files.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)})
 write(d/'PAYLOAD_INDEX.json',files)
 shape=c['case_snapshot']['parameters']['input'][0]
 after=read(d/'STATUS_AFTER.json')
 row={'baseline':'ops-math','operator':'FFTC2CFwdOp','case_id':case,'status':'verified_unregistered','reason':'Unchanged canonical checker passed all complex128 outputs through original complete aclStft system CANN AiCPU STFT; input preserved.','registered':False,'new_parent_acceptances':0,'checker':str(d/'checker.json'),'guard':str(d/'GUARD_RESULT.json'),'checker_receipt':str(d/'checker.json'),'guard_receipt':str(d/'GUARD_RESULT.json'),'guard_returncode':g['returncode'],'child_returncode':ch['returncode'],'api':c['api'],'verification':c['verification'],'seed':c['input_seed'],'full_output':{'shape':shape,'dtype':'complex128','elements':payload[0]['full_output_elements'],'raw_api_shape':shape+[1],'payload_index':str(d/'PAYLOAD_INDEX.json')},'input_preserved':True,'cleanup':cleanup,'post_health':after['health'],'post_status':after['state'],'post_reservation_owner':after['reservation'].get('metadata',{}).get('owner'),'timings':{'guard_total_wall_s':g['wall_seconds'],'checker_wall_s':c['wall_seconds'],'api_call_and_sync_wall_s':payload[0]['api_call_and_sync_wall_s'],'cpu_prepare_s':None,'queue_s':None,'compile_s':None},'native_evidence_index':str(d/'NATIVE_EVIDENCE_INDEX.json'),'source_sha256':c['source_sha256']}
 write(d/'RESULT.json',{'rows':[row]});rows.append(row)
compare=[{'path':k,'before':v,'after':sha(W/k),'unchanged':sha(W/k)==v} for k,v in read(P/'FROZEN.json').items()]
assert all(x['unchanged'] for x in compare)
write(P/'SOURCE_COMPARE_AFTER.json',compare)
write(P/'RESULT.json',{'rows':rows,'new_parent_acceptances':0,'npu_attempts':len(rows),'in_progress':len(rows)<2})
print(json.dumps([{'case_id':r['case_id'],'status':r['status'],'verification':r['verification']} for r in rows]))
if len(rows)==2:
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 write(P/'HEARTBEAT.json',{'at':now,'phase':'complete_verified_unregistered','npu_attempts':2})
 write(P/'DELIVERY.json',{'task':'fft_remaining_c128_2','worker':'604fb49cf9f9474a','finished_at':now,'status':'verified_unregistered','scope_count':2,'npu_attempts':2,'successful_case_reruns':0,'new_parent_acceptances':0,'registered':False,'result':str(P/'RESULT.json'),'findings':[{'case_id':r['case_id'],'full_elements':r['full_output']['elements'],'max_abs_err':r['verification']['max_abs_err'],'guard_rc':r['guard_returncode'],'child_rc':r['child_returncode']} for r in rows],'boundary':'Original complete ops-math aclStft; [B,N] and same-dtype unity [N] window; [B,N,1] output reshaped only; private instance ACL_COMPLEX128=17; N=hop=win, normalized=false, onesided=false, returnComplex=true. No mathematical fallback or dtype reduction. System libcpu_kernels.so executes STFT, not custom vendor libmath_aicpu_kernels.so.','checks':{'frozen_files_unchanged':len(compare),'full_output_and_input_preservation':True,'synchronization_cleanup':True,'raw_host_device_logs':'Each case ASCEND_PROCESS_LOG_PATH and NATIVE_EVIDENCE_INDEX.json','payload_hash_sizes_locations':'Each case PAYLOAD_INDEX.json','checker_snapshot_unchanged':True},'not_done':['Formal registry/ledger registration and parent acceptance (parent responsibility).','Performance benchmark/profiling (outside correctness task).','Device binary SHA256 unavailable from host; no SDK scan or extra device run.'],'remaining_issues':['Parent final review and CPU unified registration pending.','Post-run health receipts may show another worker reservation; do not interpret this as our lock leak.'],'changed_files':[str(f.relative_to(P)) for f in sorted(P.rglob('*')) if f.is_file() and f.name not in ('SCOPE.json','SPAWN.json','TASK.txt')],'shared_provider_registry_vendor_or_memory_modified':False,'notification':'Automatic supervisor delivery only.'})
 write(P/'EVIDENCE_INDEX.json',[{'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(P.rglob('*')) if f.is_file() and f.name!='EVIDENCE_INDEX.json'])
