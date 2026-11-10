from pathlib import Path
import json,hashlib,subprocess,datetime,fcntl,shutil,os
D=Path(__file__).resolve().parent;W=D.parents[1];P=D/'gdn_register1';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,j):
 p=Path(p);t=p.with_suffix('.tmp-parent');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
r=read(P/'RECOMMENDED_ACCEPT.json')['rows'][0];key=[r['baseline'],r['operator'],r['case_id']];remote={}
for field in ('checker_receipt','guard_receipt'):
 x=r[field];assert sha(P/x['path'])==x['sha256'];remote[x['original_remote_path']]=x['sha256']
c=read(P/r['checker_receipt']['path']);g=read(P/r['guard_receipt']['path']);v=c['verification']
assert c['status']=='source_changed' and g['returncode']==1 and v['checked_outputs']==v['total_outputs']==2 and not v['unchecked_reason']
assert c['inputs_preserved'] and not c['oracle_source_changed'] and not c.get('sync_error') and not c.get('cleanup_error') and not c.get('error')
assert c['input_seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big')
e=read(P/'attempt_001/SOURCE_EQUIVALENCE.json');changes={p:{'before':c['provider_source_sha256'].get(p),'after':c['provider_source_sha256_after'].get(p)} for p in c['provider_source_sha256'].keys()|c['provider_source_sha256_after'].keys() if c['provider_source_sha256'].get(p)!=c['provider_source_sha256_after'].get(p)};assert changes==e['changes'] and len(changes)==3
for h,x in e['snapshots'].items():assert hashlib.sha256(x['text'].encode()).hexdigest()==h;assert sha(P/x['evidence'])==h
init=changes['TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py'];bt=e['snapshots'][init['before']]['text'];at=e['snapshots'][init['after']]['text'];assert at.startswith(bt) and 'GDNFwdOp' not in at[len(bt):]
for p,change in changes.items():assert p.endswith('__init__.py') or 'transformer_attention_misc_' in p
checks=read(P/'attempt_001/CHECKS.json');assert checks['contract']['params']==c['case_snapshot']['parameters']
for x in checks['critical']:
 assert sha(P/x['evidence'])==x['sha256']
 if 'tested' in x:
  assert x['receipt_hash']==x['sha256']==c['source_sha256'][x['tested']];remote[x['tested']]=x['sha256'];remote[x['frozen']]=x['sha256']
 else:remote[x['path']]=x['sha256']
n=read(P/'attempt_001/NATIVE.json')
for x in n['selected_pins']:
 assert c['source_sha256'][x['path']]==x['sha256']==x['current_sha256'];remote[x['path']]=x['sha256']
assert len(n['loaded_opapi'])==1;loaded=n['loaded_opapi'][0];assert loaded['path']==n['recipe']['opapi'] and remote[loaded['path']]==loaded['sha256']
# Verify historical loaded-library capture is the original case's retained record.
lp=P/'attempt_001/evidence'/r['checker_receipt']['original_remote_path'].lstrip('/');lp=lp.with_name('loaded-libraries.json');assert loaded in read(lp);remote[str(lp).split('/evidence',1)[1]]=sha(lp)
detail=read(P/'attempt_003_native/NATIVE_DETAIL.json')
for x in detail['kernel_source_equivalence']:
 assert x['equal'] and x['sha256']==x['original_sha256'];remote[x['installed']]=x['sha256'];remote[x['original']]=x['sha256']
reg=read(P/'REGISTRATION.json');remote[reg['factory']]=reg['factory_sha256'];remote.update(reg['binding_hashes'])
# Same frozen binding/direct transport and primary original binary criteria as52 earlier accepted GDN keys.
existing=next(x for x in read(D/'ACCEPTED.json')['rows'] if x['key'][:2]==key[:2]);prior=read(existing['checker'])
for name in ('binding.py','direct_acl.py','entry.py'):
 old=[h for p,h in prior['source_sha256'].items() if p.endswith('/chunk_package/'+name)];new=[h for p,h in c['source_sha256'].items() if p.endswith('/chunk_package/'+name)];assert old==new and old
assert prior['source_sha256'][loaded['path']]==loaded['sha256']
src=(W/'TileOPs/scripts/ascend_pair_check.py').read_text();assert sha(W/'TileOPs/scripts/ascend_pair_check.py')==c['verifier_sha256'];assert src.index('torch.npu.synchronize()',src.index('    finally:'))<src.index('args.output.write_text',src.index('    finally:'))
# Reuse reviewed CPU-only registrar setup, omitting every mutation/preflight override.
setup=(P/'register.py').read_text().split('# CPU preflight actual registry')[0].replace("A.mkdir(exist_ok=False)","None")
pre='import pathlib,hashlib\nfor p,h in '+repr(remote)+'.items():\n assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p\n'
end='''
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert cid in json.loads(reg.read_text())
 candidate=registry.candidate_factory(case,'ops-transformer')()
 assert candidate.api=='ops-transformer::aclnnChunkGatedDeltaRule'
print(json.dumps({'api':candidate.api,'actual_binding':candidate.implementation.fn.__code__.co_filename,'events':events,'hashes_verified':HASHCOUNT,'device_runs':0,'writes':0}))
'''.replace('HASHCOUNT',str(len(remote)))
p=subprocess.run([str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'),'openlibing-DevEnv-055409','python3 -S -'],input=pre+setup+end,text=True,capture_output=True,timeout=50);assert p.returncode==0,p.stderr;proof=json.loads(p.stdout);put(D/'PARENT_GDN_ADDITIONAL1_LIVE_CHECK.json',proof)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();rp=D/'PARENT_GDN_ADDITIONAL1_REVIEW.json';review={'at':now,'key':key,'accepted_added':0,'full_outputs':v,'source_equivalence':'Exactly3 provider changes: init appends unrelated GQA/decode routes and two new attention modules. GDN call chain unchanged.','original_boundary':'Whole original ChunkGatedDeltaRule via explicit absolute original opapi path. Historical receipt fingerprints binding, entry and direct ACL bridge plus actual opapi and runtime artifacts; retained loaded-library record matches. Byte-identical frozen call chain and primary binary match52 previously accepted GDN keys. No math fallback; full output and final_state checked.','proof_limit':'No standalone per-case device launch/build log retained. Attribution uses historically fingerprinted complete original call chain, loaded opapi identity and pinned original artifacts, not a claimed observed launch or another cases launch. Unlike heldFP8five, direct bridge is historically fingerprinted.','guard_returncode_preserved':1,'checker_status_preserved':'source_changed','original_kernel_sources':len(detail['kernel_source_equivalence']),'remote_hashes_verified':len(remote),'new_npu_runs':0,'live_registry_proof':str(D/'PARENT_GDN_ADDITIONAL1_LIVE_CHECK.json')};put(rp,review)
with open(str(D/'ACCEPTED.json')+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(D/'ACCEPTED.json');by={tuple(x['key']):x for x in ledger['rows']};before=len(by)
 if tuple(key) not in by:by[tuple(key)]={'at':now,'key':key,'accepted':True,'host':'server2','checker':str((P/r['checker_receipt']['path']).resolve()),'guard':str((P/r['guard_receipt']['path']).resolve()),'full_outputs':v,'parent_review':str(rp),'qualification':review['proof_limit'],'original_checker_status':'source_changed','guard_returncode':1}
 ledger.update(rows=list(by.values()),updated_at=now);put(D/'ACCEPTED.json',ledger)
review.update(accepted_added=len(by)-before,accepted_total=len(by));put(rp,review);old=P/'PARENT_REVIEW.json';shutil.copy2(old,P/('PARENT_REVIEW.before_final_'+now.replace(':','-')+'.json'));put(old,review)
s=read(D/'STATE.json');oldq=D/'LINEAR_ROOT_QUEUED29.json';newq=D/'LINEAR_ROOT_QUEUED28_AFTER_GDN1.json';q=read(oldq);remaining=[x for x in q if [x['baseline'],x['operator'],x['case_id']]!=key];assert len(remaining)==28;put(newq,remaining);s['root_queued_scopes']=[str(newq) if x==str(oldq) else x for x in s['root_queued_scopes']];s['root_queued_count']=sum(len(read(x)) for x in s['root_queued_scopes']);s.setdefault('parent_accepted_additional_keys',[])
if key not in s['parent_accepted_additional_keys']:s['parent_accepted_additional_keys'].append(key)
s.update(completed=len(by),parent_accepted_completed=len(by),last_status_accepted=len(by),last_status_at=now,last_acceptance_review=str(rp));s['ownership_audit'].update(accepted=len(by),root_queued=s['root_queued_count']);put(D/'STATE.json',s)
M=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(M)+'.lock','a') as a,open(M.parent/'STATE.lock','a') as b:
 fcntl.flock(a,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(M,M.with_name('STATE.before_gdn_additional1_'+now.replace(':','-')+'.json'));j=read(M);j['integration_700_20261026'].update({k:s[k] for k in ['parent_accepted_completed','last_acceptance_review','last_status_at','root_queued_scopes','root_queued_count','parent_accepted_additional_keys','ownership_audit']});put(M,j)
print(json.dumps({'accepted_added':review['accepted_added'],'accepted_total':len(by),'remote_hashes_verified':len(remote),'npu_runs':0}))
