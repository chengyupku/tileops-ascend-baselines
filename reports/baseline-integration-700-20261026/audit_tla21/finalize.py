import json,pathlib,hashlib,subprocess,ast,yaml,datetime
D=pathlib.Path(__file__).resolve().parent;W=D.parents[2];x=json.loads((D/'REMOTE_EVIDENCE.json').read_text());files=x['files']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def file(s):return next(v for p,v in files.items() if p.endswith(s))
def ref(v):return {k:v[k] for k in ['path','sha256']}
def lines(v,needle):return [dict(line=i+1,text=t) for i,t in enumerate(v['text'].splitlines()) if needle in t]
bg=file('/gemm_rest_pto.guard.json');batch=json.loads(bg['text']);log=file('/gemm_rest_pto.guard.log');runner=file('/run_batch9.py');adapter=file('/frozen_gemm_rest7/adapter.py');jit=file('/tilelang/jit/__init__.py');lower=file('/tilelang/engine/lower.py')
assert batch['returncode']==0
assert 'external NPU' not in log['text']
original=[]
for s in ['examples/gemm/example_gemm.py','examples/batch_gemm/batch_gemm.py']:
 v=file('/'+s);p=subprocess.run(['git','-C',str(W/'ascend_baselines/tilelang-ascend'),'show','HEAD:'+s],text=False,capture_output=True)
 original.append(dict(source=ref(v),git_command=['git','show','HEAD:'+s],returncode=p.returncode,stderr=p.stderr.decode(),git_blob_sha256=hashlib.sha256(p.stdout).hexdigest(),matches=p.returncode==0 and hashlib.sha256(p.stdout).hexdigest()==v['sha256']))
assert all(v['matches'] for v in original)
conflict=dict(code='COMPILER_TARGET_CHANGED',description="Frozen adapter compile_pto forces target='pto', but original matmul/batch_matmul decorators omit target and retain jit default 'auto'. Runtime lower.py selects target.build.tilelang_ascend for auto and target.build.tilelang_ascend_pto for pto. Calls receipts confirm target pto. Violates assigned no compiler change rule.",adapter=dict(**ref(adapter),evidence=lines(adapter,"target='pto'")),jit_default=dict(**ref(jit),evidence=lines(jit,'target: str | Target = "auto"')),codegen=dict(**ref(lower),evidence=lines(lower,'target.model')+lines(lower,'target.build.tilelang_ascend')),original_decorators=[dict(**v['source'],evidence=lines(files[v['source']['path']],'@tilelang.jit')) for v in original])
rows=[]
for r in x['rows']:
 k=r['key'];c=r['checker']['data'];g=r['child_guard']['data'];params=c['case_snapshot']['parameters'];dtype=params['a'][1]
 spec=file('/gemm_bmm.yaml' if k[1]=='BmmFwdOp' else '/gemm.yaml');workloads=yaml.safe_load(spec['text'])[k[1]]['workloads']
 w=next(w for w in workloads if w['label']+'-'+dtype==k[2]);assert any(dc['T']==dtype for dc in w['dtype_cases'])
 if k[1]=='BmmFwdOp':expected={'a':[[w['B'],w['M'],w['K']],dtype],'b':[[w['B'],w['K'],w['N']],dtype],'d':[[w['B'],w['M'],w['N']],dtype]}
 else:
  ta=w.get('trans_a',False);tb=w.get('trans_b',False)
  expected={'a':[[w['K'],w['M']] if ta else [w['M'],w['K']],dtype],'b':[[w['N'],w['K']] if tb else [w['K'],w['N']],dtype],'d':[[w['M'],w['N']],dtype],'trans_a':ta,'trans_b':tb}
 assert params==expected
 seed=int.from_bytes(hashlib.sha256(f'{k[1]}::{k[2]}'.encode()).digest()[:4],'big');assert seed==c['input_seed']
 assert c['verifier_sha256']==file('/scripts/ascend_pair_check.py')['sha256']
 assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed']
 assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert g['returncode']==0 and batch['started']<=g['started']<g['finished']<=batch['finished'] and k[1:] in batch['keys']
 assert not r['issues'];assert r['route']['key']==k and r['route']['params']==params
 assert c['case_snapshot']['declared_output_names']==['d']
 for kind in ['candidate','reference']:
  obs=c['observed_calls'][kind];assert len(obs)==1 and obs[0]['returned'];out=obs[0]['outputs'];assert out['shape']==params['d'][0] and out['dtype']==dtype
  assert all(t['shape']==params[n][0] and t['dtype']==params[n][1] and t['layout']=='torch.strided' for t,n in zip(obs[0]['inputs']['items'],['a','b']))
 assert all(v['data']['successful_host_launches']>0 for v in r['calls'])
 for v in r['calls']:
  for call in v['data']['calls']:
   assert call['target']=='pto'
   for name in ['binary','generated_source','adapter_source']:
    path=call[name]['path'];path=path if path.startswith('/') else '/opt/tileops-all-pairs/workspace/'+path
    assert r['hashes'][path]['actual']==call[name]['sha256']
 rows.append(dict(key=k,accept_recommended=False,status='REJECT_COMPILER_TARGET_CHANGE',issues=[conflict['code']],checker=ref(r['checker']),guard=dict(**ref(bg),returncode=batch['returncode'],scope='complete 21-case shared batch',log=ref(log)),child_guard=dict(**ref(r['child_guard']),returncode=g['returncode'],scope='case checker plus explicit synchronization'),formal_module=x['formal_module'],formal_factory='benchmarks.ascend_all_pairs.tla.integration700_tla_simple_tilelang_gemm_rest7:factory',factory='benchmarks.ascend_all_pairs.tla.integration700_tla_simple_tilelang_gemm_rest7:factory',tested_registered_identity=dict(tested_adapter=ref(file('/gemm_adapter_pto.py')),registered_adapter=ref(adapter),same_sha256=True,private_runtime_pins_verified=True),provider_route=dict(provider=x['provider'],registry=ref(file('/registry.py')),entry='candidate_factory(case, library)()',output=r['route'],wrong_parameters_rejected=True,stub_boundary='only formal _prepare device construction; real candidate_factory, provider dispatch, matches and factory parameter validation executed',device_imported=False),canonical=dict(manifest=ref(spec),parameters=params,seed=seed,seed_protocol=ref(file('/baseline_protocol.py'))),verification=dict(**c['verification'],inputs_preserved=True,oracle_source_changed=False,tolerance={'rtol':0.001 if dtype=='float16' else 0.016,'atol':0.001 if dtype=='float16' else 0.016},state='stateless matmul; only output d; a and b preserved',observed_calls=c['observed_calls']),source_provenance=dict(actual_checked_hashes=r['hashes'],native_launch=r['calls'],generated_kernels=r['kernels'],original_source_git_verified=original,original_vendor_commit='1d5daf0385156cbe0c499a16c10935c5190642a4',configuration=[128,256,64],binding='complete original builders; public transpose/contiguous/pad/crop only; no arithmetic fallback',compiler_target='pto',compiler_conflict=conflict['code']),synchronization=dict(runner=ref(runner),evidence=lines(runner,'synchronize()')),raw_evidence=dict(path=str(D/'REMOTE_EVIDENCE.json'),sha256=sha(D/'REMOTE_EVIDENCE.json'))))
report=dict(schema='baseline700-audit/v1',audited_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope_count=21,recommended_count=0,fully_verified_numerics_and_route_count=21,NPU_commands=0,device_imports=0,rows=rows,blocking_finding=conflict)
(D/'RECOMMENDED_ACCEPT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
(D/'COMPILER_TARGET_FINDING.json').write_text(json.dumps(conflict,indent=2))
(D/'AUDIT_SUMMARY.json').write_text(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
print(json.dumps(dict(rows=len(rows),recommended=0,hash_mismatches=0,canonical_verified=21,real_registry_routes=21,guard_returncode=batch['returncode'],blocker=conflict['code'])))
