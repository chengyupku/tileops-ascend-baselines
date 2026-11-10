from pathlib import Path
import json,hashlib,datetime,subprocess,fcntl,os,shutil
D=Path(__file__).resolve().parent;R=D/'audit_tla21';E=json.loads((R/'REMOTE_EVIDENCE.json').read_text());F=json.loads((R/'COMPILER_TARGET_FINDING.json').read_text());A=json.loads((D/'ACCEPTED.json').read_text());accepted={tuple(x['key']) for x in A['rows']};keys=[];targets=[];digests={}
for x in E['rows']:
 key=x['key'];assert tuple(key) not in accepted;keys.append(key)
 p=Path(x['checker']['path'].replace('/opt/tileops-all-pairs/workspace','/root/dyq/workspace',1));data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==x['checker']['sha256'];c=json.loads(data)
 assert c['status']=='verified' and c['inputs_preserved'] and c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert not c['provider_source_changed'] and not c['oracle_source_changed']
 assert c['input_seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big')
 assert [c['library'],c['op'],c['case_id']]==key
 assert x['calls']
 for call in x['calls']:
  assert call['data']['successful_host_launches']>0;digests[call['path']]=call['sha256']
  for launch in call['data']['calls']:assert launch['target']=='pto';targets.append({'key':key,'target':launch['target'],'calls_file':call['path'],'calls_sha256':call['sha256']})
for field in ['adapter','jit_default','codegen']:
 v=F[field];snap=E['files'][v['path']];assert hashlib.sha256(snap['text'].encode()).hexdigest()==v['sha256']==snap['sha256'];digests[v['path']]=v['sha256']
for v in F['original_decorators']:digests[v['path']]=v['sha256']
assert len(keys)==21
code='import pathlib,hashlib,json\nhashes='+repr(digests)+'\nfor p,h in hashes.items():\n assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p\nprint(json.dumps({"current_hashes_verified":len(hashes),"device_imports":False}))\n'
p=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-661495','python3 -S -'],input=code,text=True,capture_output=True,timeout=40);assert p.returncode==0,p.stderr
now=datetime.datetime.now(datetime.timezone.utc).isoformat();report={'at':now,'job':'48749ad5f2b74c16','scope':21,'accepted_added':0,'previously_in_accepted':0,'ledger_total_unchanged':len(accepted),'numerical_complete':21,'decision':'withhold_acceptance_compiler_target_changed','finding':F,'parent_direct_checks':{'checker_hashes_and_complete_outputs':21,'native_call_targets':targets,'remote_current_hash_check':json.loads(p.stdout)},'impact_review_worker':'beeb5a1dc276447a','impact_review_directory':str(D/'tla_compiler_boundary44'),'next':'Read-only44 scope audit already active, including prior23 accepted; do not duplicate/retry NPU or alter registry until exact affected keys established.','npu_runs':0,'evidence_note':'Batch_gemm captured text omits final newline; verified true raw source hash remotely instead of treating truncated display as source bytes.','time_waste_correction':'Requested actual UTC JSON heartbeat; worker plain HEARTBEAT contained stale04:48 timestamp. Native process identity remains primary survival measure.'}
(R/'PARENT_REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
M=Path('/root/dyq/workspace/.agent_memory/THREADS/baseline-inspector-20261023/STATE.json')
with open(str(M)+'.lock','a') as l1,open(M.parent/'STATE.lock','a') as l2:
 fcntl.flock(l1,fcntl.LOCK_EX);fcntl.flock(l2,fcntl.LOCK_EX);j=json.loads(M.read_text());shutil.copy2(M,M.with_name('STATE.before_tla21_review_'+now.replace(':','-')+'.json'));j['integration_700_20261026']['tla21_parent_review']=str(R/'PARENT_REVIEW.json');t=M.with_suffix('.tmp-parent');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');os.replace(t,M)
print(json.dumps({'accepted_added':0,'ledger_total':len(accepted),'direct_checkers':21,'native_calls':len(targets),'remote_current_hashes':len(digests)}))
