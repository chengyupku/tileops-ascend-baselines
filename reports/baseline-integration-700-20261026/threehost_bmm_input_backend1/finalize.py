from control import *
import hashlib
D=P/'server1';C=D/'square-4k-bfloat16'
def read(p):return json.loads(p.read_text())
def dump(n,d):(P/n).write_text(json.dumps(d,indent=2)+'\n')
assert read(P/'LOCAL_VERIFICATION.json')['all_hashes_match']
checker=read(C/'representative.json');cap=read(C/'CAPTURE.json');diag=read(D/'DIAGNOSTIC.json');guard=read(D/'GUARD_FINAL.json');native=read(C/'NATIVE_CALLS.json')
input_groups={str(i):{n:v['sha256'] for n,v in cap.items() if n.endswith('input'+str(i))} for i in [0,1]}
assert all(len(set(v.values()))==1 for v in input_groups.values())
assert checker['input_seed']==1868357590 and checker['status']=='failed' and checker['stage']=='verify'
assert guard['exit_code']==1 and guard['phase']=='finished'
assert [n['return_code'] for n in native if n['event']=='return']==[0,0]
assert diag['original_settings']==diag['restored_settings']
dump('DIAGNOSTIC.json',dict(diag,interpretation={'established':'Same captured inputs show CPU BF16 torch.bmm output depends on public mkldnn backend selection in this fixed output window. Default oneDNN gemm:acl window output exactly matches captured full default reference. Disabled-mkldnn window agrees with native within original tolerance.','not_established':['Exact internal multiply/accumulation precision or implementation defect','Native correctness for full output under an approved reference setting','Full-size disabled-mkldnn runtime or completion within budget','Byte equality to historical run (only new run bytes are known)'],'next':'Parent reviews whether public mkldnn.enabled=False preserves original runtime contract, then separately decides on a bounded full original checker. No new checker was run after this diagnostic.'}))
dump('INPUTHASH.json',dict(seed=checker['input_seed'],old_run_byte_equality_claim=False,new_run_input_groups=input_groups,unique_input_bytes=268435456,byteorder='little',dtype='torch.bfloat16',shape=[4,4096,4096],exact_bytes_directory=str(C)))
backend_lines=[l for l in (D/'guard.log').read_text().splitlines() if l.startswith(('onednn_verbose','CPU_REFERENCE_'))]
(P/'BACKEND.log').write_text('\n'.join(backend_lines)+'\n')
(P/'CPU_REFERENCE_SETTINGS.json').write_bytes((C/'CPU_REFERENCE_SETTINGS.json').read_bytes())
delivery=dict(status='completed_capture_and_bounded_cpu_diagnostic_fullchecker_failed',accepted_added=0,key=['ops-transformer','BmmFwdOp','square-4k-bfloat16'],host='server1',remote_report=R,single_guarded_fullchecker=True,guard_exit=guard['exit_code'],guard_owner=guard['owner'],guard_requested_at=guard['requested_at'],guard_finished_at=guard['updated_at'],setup_failures=0,external_interference=False,checker_status=checker['status'],checker_error=checker['error'],native_return_codes=[0,0],source_frozen_unchanged=read(C/'EXECUTION.json')['frozen_unchanged'],seed=checker['input_seed'],input_bytes_saved=268435456,input_same_run_preserved=True,not_claimed_same_as_old_run=True,reference_settings=str(C/'CPU_REFERENCE_SETTINGS.json'),full_reference_primitive='oneDNN3.7.1 cpu matmul gemm:acl BF16/BF16/BF16, 4x4096x4096:4x4096x4096, attr-scratchpad:user',diagnostic=str(P/'DIAGNOSTIC.json'),diagnostic_window=diag['window'],diagnostic_comparisons=diag['comparisons'],cpu_diagnostic_exit=int((D/'cpu_backend.exit').read_text()),cpu_isolation='-I -S -B; backend autoload disabled; no .pth loading; import/access guards; no torch_npu/Ascend ACL mappings or device handles before/after; 120s wall, 8GiB AS, 110s CPU',cpu_original_settings_restored=True,checks=['CPU preflight paths/new output/cgroupv1 raw capacity/disk/aarch64 ABI','Unchanged binding.py, adapter.py and original checker/workload frozen closure','Captured candidate/reference pre/post and actual CPU inputs have matching hashes','All transferred sealed files SHA256 verified locally','Native complete API workspace and launch return 0; full checker numerical failure preserved'],verification=read(P/'LOCAL_VERIFICATION.json'),remaining=['Parent contract review required before public backend setting can be used in formal full-case verification','No claim of corrected fullcase, native numerical validity, or precise backend accumulation cause','Historical inputs absent; new seed is same integer only'],changed_files_scope='Only this local report and remote same-name private report. No shared source/registry/ledger/memory changes.',evidence={'guard':str(D/'guard.log'),'guard_final':str(D/'GUARD_FINAL.json'),'checker':str(C/'representative.json'),'native':str(C/'NATIVE_CALLS.json'),'native_closure':str(D/'NATIVE_CLOSURE.json'),'inputhash':str(P/'INPUTHASH.json'),'settings':str(C/'CPU_REFERENCE_SETTINGS.json'),'backend':str(P/'BACKEND.log'),'cpu_backend':str(D/'cpu_backend.log')})
dump('DELIVERY.json',delivery)
(P/'DIAGNOSTIC_REVIEW.json').write_bytes((P/'DIAGNOSTIC.json').read_bytes())
for n in ['DIAGNOSTIC_REVIEW.json','INPUTHASH.json','DELIVERY.json','BACKEND.log']:upload(n)
hb('complete_accepted_added_0')
pins={}
for p in P.rglob('*'):
 if p.is_file() and p.name not in ['FILEHASHES.json','server1_evidence.tar.gz']:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  pins[str(p.relative_to(P))]=h.hexdigest()
dump('FILEHASHES.json',pins)
print(json.dumps(dict(status=delivery['status'],accepted_added=0,verified_files=delivery['verification'],comparisons=diag['comparisons']),indent=2))
