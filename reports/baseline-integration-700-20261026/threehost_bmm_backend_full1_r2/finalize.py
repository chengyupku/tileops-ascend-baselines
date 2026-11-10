from control import *
import hashlib
D=P/'server1';C=D/'square-4k-bfloat16'
def read(p):return json.loads(p.read_text())
def dump(n,d):(P/n).write_text(json.dumps(d,indent=2)+'\n')
assert read(P/'LOCAL_VERIFICATION.json')['all_hashes_match']
checker=read(C/'representative.json');cap=read(C/'CAPTURE.json');guard=read(D/'GUARD_FINAL.json');native=read(C/'NATIVE_CALLS.json');variant=read(C/'BACKEND_VARIANT.json');execution=read(C/'EXECUTION.json')
groups={str(i):{n:v['sha256'] for n,v in cap.items() if n.endswith('input'+str(i))} for i in [0,1]}
assert all(len(set(v.values()))==1 for v in groups.values())
assert checker['input_seed']==1868357590
assert [n['return_code'] for n in native if n['event']=='return']==[0,0]
assert execution['frozen_unchanged']
events={x['phase']:x for x in variant['events']}
assert all(x['bmm_unchanged'] for x in variant['events'])
assert events['reference_before']['mkldnn_enabled']==events['reference_after']['mkldnn_enabled']==False
assert events['restored']['mkldnn_enabled']==variant['original']
assert cap['reference_0_output']['shape']==[4,4096,4096]
dump('INPUTHASH.json',dict(seed=checker['input_seed'],old_run_byte_equality_claim=False,new_run_input_groups=groups,unique_input_bytes=268435456,dtype='torch.bfloat16',shape=[4,4096,4096],exact_bytes_directory=str(C)))
delivery=dict(status='completed_fullchecker_'+checker['status'],accepted_added=0,key=['ops-transformer','BmmFwdOp','square-4k-bfloat16'],host='server1',remote_report=R,single_guarded_fullchecker=True,guard_exit=guard['exit_code'],guard_owner=guard['owner'],guard_requested_at=guard['requested_at'],guard_finished_at=guard['updated_at'],checker_status=checker['status'],checker_error=checker.get('error'),native_return_codes=[0,0],source_frozen_unchanged=True,seed=checker['input_seed'],input_same_run_preserved=True,not_claimed_same_as_old_run=True,backend_variant='torch.backends.mkldnn.enabled=False',backend_restored=True,original_default_failure_preserved=str(P.parent/'threehost_bmm_input_backend1/DELIVERY.json'),output_elements=67108864,timings=execution['timings'],checker_wall_seconds=checker['wall_seconds'],cpu_reference_seconds=events['reference_after']['time']-events['reference_before']['time'],verification_result=checker['verification'],checks=['CPU capacity/disk/aarch64 ELF/runtime dependency preflight','Private original checker/candidate/workload/kernel closure byte-identical before run and frozen unchanged afterwards','Native actual pointers/maps/inodes and workspace/launch return codes recorded','Reference original torch.bmm identity and callable source recorded before/after; public backend restored in finally','Captured inputs preserved through candidate and reference','Local transferred sealed file hashes verified'],verification=read(P/'LOCAL_VERIFICATION.json'),remaining=['Formal registration by parent; accepted_added remains 0','New run input hashes recorded; no historical byte-equality claim'],changed_files_scope='Only this local task report and remote same-name directory; shared source/ledger/memory untouched',evidence={n:str(v) for n,v in dict(checker=C/'representative.json',guard=D/'guard.log',native=C/'NATIVE_CALLS.json',native_closure=D/'NATIVE_CLOSURE.json',backend_variant=C/'BACKEND_VARIANT.json',inputhash=P/'INPUTHASH.json',execution=C/'EXECUTION.json',frozen=D/'FROZEN.json',device_binary=P/'device_proof/PROOF.json').items()})
dump('DELIVERY.json',delivery);dump('RESULT.json',delivery)
hb('complete_accepted_added_0')
for n in ['INPUTHASH.json','DELIVERY.json','RESULT.json','HEARTBEAT.json']:upload(n)
pins={}
for f in P.rglob('*'):
 if f.is_file() and f.name not in ['FILEHASHES.json','server1_evidence.tar.gz']:
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
  pins[str(f.relative_to(P))]=h.hexdigest()
dump('FILEHASHES.json',pins);upload('FILEHASHES.json')
print(json.dumps(delivery,indent=2))
