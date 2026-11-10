import pathlib,json,hashlib,datetime,sqlite3
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'fft_remaining_c64_7';read=lambda p:json.loads(p.read_text());cache={}
def sha(p):
 p=pathlib.Path(p)
 if str(p) not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  cache[str(p)]=h.hexdigest()
 return cache[str(p)]
for x in read(P/'EVIDENCE_INDEX.json'):assert sha(P/x['path'])==x['sha256'],x['path']
for p,h in read(P/'FROZEN.json').items():assert sha(P/p)==h,p
assert sha(P/'ascend_pair_check.snapshot.py')==sha(W/'TileOPs/scripts/ascend_pair_check.py')
rows=read(P/'RESULT.json')['rows'];assert len(rows)==7
sc={(r['baseline'],r['operator'],r['case_id']) for r in read(P/'SCOPE.json')};assert sc=={(r['baseline'],r['operator'],r['case_id']) for r in rows};native_count=0;timings=[]
for r in rows:
 C=P/r['case_id'];c=read(C/'checker.json');g=read(C/'GUARD_RESULT.json');gm=read(C/'GUARD_METADATA.json');assert c['status']=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']==1 and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 assert c['input_seed']==int.from_bytes(hashlib.sha256(('FFTC2CFwdOp::'+r['case_id']).encode()).digest()[:4],'big')
 assert g['returncode']==read(C/'CHILD_RESULT.json')['returncode']==0;assert gm['owner']=='baseline700-20261026/fft_remaining_c64_7';assert gm['acquired_at']<c['started_at']
 for field in ['source_sha256','oracle_source_sha256']:
  for p,h in c[field].items():assert sha(W/p)==h,p
 for x in read(C/'PAYLOAD_INDEX.json'):assert sha(x['path'])==x['sha256']
 pay=read(C/'PAYLOADS.json');assert len(pay)==1 and pay[0]['input_preserved'] and pay[0]['full_output_elements']==r['payloads'][0]['full_output_elements'];assert read(C/'WINDOW_BINDING.json')['all_one_plus_zero_j']
 raw=[]
 for x in read(C/'NATIVE_EVIDENCE_INDEX.json'):
  assert sha(x['path'])==x['sha256'];ls=pathlib.Path(x['path']).read_text(errors='replace').splitlines();selected=[line for line in ls if any(t in line for t in ['RunCpuKernel[STFT]', 'windowOptional: aclTensor','RegisterCpuKernel:','Load so success','LoadBinaryFromJson','start to Launch STFT','AicpuCCTask::Run','aclStft path=AiCpu'])];native_count+=len(selected);raw.extend(selected)
 joined='\n'.join(raw);assert 'RunCpuKernel[STFT] success' in joined and 'windowOptional: aclTensor' in joined and 'libcpu_kernels.so' in joined and 'DT_COMPLEX64' in joined
 assert f'({gm["child"]["pid"]},python)' in joined
 timings.append({'case_id':r['case_id'],'guard_wall_s':r['timings']['guard_total_wall_s'],'queue_s':(datetime.datetime.fromisoformat(gm['acquired_at'])-datetime.datetime.fromisoformat(gm['requested_at'])).total_seconds(),'full_output_elements':r['payloads'][0]['full_output_elements'],'max_abs_err':c['verification']['max_abs_err']})
qual=read(D/'fft_window_first1/PARENT_REVIEW.json')['qualification']
review={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'eligible_pending_registration','keys':sorted(sc),'hashes_checked':len(cache),'native_selected_raw_lines_checked':native_count,'canonical_checks':7,'successful_case_reruns':0,'parent_device_runs':0,'timings':timings,'qualification':qual,'boundary':'Same original complete aclStft, explicit same-dtype unity window, batched axis retained, same complex64 dtype. Frozen adapter reviewed; no newmath or dtype reduction.','registration_wait':'All seven checks finished; wait native completion then parent registration.'}
(P/'PARENT_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review))
