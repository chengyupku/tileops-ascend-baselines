import pathlib,json,hashlib,re,datetime,yaml,difflib
A=pathlib.Path(__file__).resolve().parent;W='/opt/tileops-all-pairs/workspace/';D=W+'reports/baseline-integration-700-20261026/transformer_attention_misc/';I=json.loads((A/'REMOTE_INDEX.json').read_text());E=json.loads((A/'EXTRA.json').read_text());F=json.loads((A/'FINAL_REMOTE.json').read_text());N=json.loads((A/'NATIVE_LAUNCH.json').read_text());scope=json.loads((A/'SCOPE.json').read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
# Materialize all inspected text into private evidence, retaining original remote identity.
for origin,item in list(E['files'].items())+list(F['files'].items()):
 remote=origin if origin.startswith('/') else W+origin;p=A/'remote_files'/remote.removeprefix(W);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['text']);assert sha(p)==item['sha256'];I['files'][remote]={'sha256':item['sha256'],'local_copy':str(p)}
def ref(p):
 p=p if p.startswith('/') else W+p
 v=I['files'][p];return {'path':v['local_copy'],'remote_path':p,'sha256':v['sha256']}
def read(p):return json.loads(pathlib.Path(ref(p)['path']).read_text())
def verify_map(m):
 out=[]
 for p,h in m.items():
  x=ref(p);assert x['sha256']==h,(p,h,x);out.append(x)
 return out
manifest=yaml.safe_load(E['files']['TileOPs/src/tileops/manifest/spec/attention.yaml']['text'])
allcritical=I['critical']+F['critical'];assert all(v['sha256']==v['expected'] for v in allcritical)
(A/'CRITICAL_SOURCE_HASHES.json').write_text(json.dumps(allcritical,indent=2));rows=[]
for s,route in zip(scope,I['routes']):
 key=[s['baseline'],s['operator'],s['case_id']];c=read(D+s['checker_receipt']);g=read(D+s['guard_receipt']);issues=[]
 assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed'];assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 seed=int.from_bytes(hashlib.sha256(f"{s['operator']}::{s['case_id']}".encode()).digest()[:4],'big');assert c['input_seed']==seed
 formal=ref(route['formal_module']);assert formal['sha256']==s['factory_sha256'];tested=[(p,h) for p,h in c['source_sha256'].items() if 'integration700_transformer_attention_misc_' in p];assert len(tested)==1 and tested[0][1]==formal['sha256'];assert route['exact'] and route['key']==key
 assert ref('TileOPs/scripts/ascend_pair_check.py')['sha256']==c['verifier_sha256']
 oracle=verify_map(c['oracle_source_sha256']);sources=verify_map(c['source_sha256'])
 op=manifest[s['operator']];dt=s['case_id'].rsplit('-',1)[1];wl=next(w for w in op['workloads'] if w['label']+'-'+dt==s['case_id']);assert {'T':dt} in wl['dtype_cases'];env={k:v for k,v in wl.items() if k not in ('label','dtype_cases')};env['T']=dt
 params={k:v['default'] for k,v in op['signature'].get('params',{}).items() if 'default' in v};params.update({k:v for k,v in env.items() if k in op['signature'].get('params',{})});env.update(params)
 for name,decl in {**op['signature']['inputs'],**op['signature']['outputs']}.items():
  dims=eval(decl['shape'],{'__builtins__':{}},env);dtype=env.get(decl['dtype'],decl['dtype']);assert c['case_snapshot']['declared_tensors'][name]==[dims,dtype],(key,name)
 for name,v in params.items():assert c['case_snapshot']['parameters'][name]==v,(key,name)
 lo=c['started_at'][:19].replace('T','-');hi=c['finished_at'][:19].replace('T','-');launch=[]
 for log in N:
  for line in log['matches']:
   txt=line['text'];t=re.search(r'2026-10-26-\d\d:\d\d:\d\d',txt)
   if 'StreamLaunchKernelV1:' in txt and t and lo<=t[0]<=hi:launch.append({'path':log['path'],'sha256':log['sha256'],**line})
 if not launch:issues.append('No actual StreamLaunchKernelV1 in checker time interval')
 kernels=set(re.search(r'kernel_name=(te_flashattentionscore_[a-f0-9]+)',v['text'])[1] for v in launch)
 ownership=[]
 for log in N:
  hits=[t for t in log['matches'] if 'classicpilot_attention_transformer' in t['text'] and (any(k in t['text'] for k in kernels) or 'flash_attention_score' in t['text'])]
  # same process log or compile log naming exact launched kernel; retain compact records
  for t in hits:
   if any(k in t['text'] for k in kernels):ownership.append({'path':log['path'],'sha256':log['sha256'],**t})
 if not ownership:issues.append('No compile source ownership joined to launched kernel')
 guardlog=ref(str(pathlib.PurePosixPath(D+s['guard_receipt']).with_name('guard.log')));logtxt=pathlib.Path(guardlog['path']).read_text();assert not any(v in logtxt.lower() for v in ['interference detected','external npu process detected']);assert g['returncode'] in (0,124)
 before=g['end']-datetime.datetime.fromisoformat(c['finished_at']).timestamp();assert before>0
 started=read(D+'runs/'+s['case_id']+'/started.json');assert started['guard']['phase']=='running';assert started['guard']['owner']=='baseline700-20261026/transformer_attention_misc'
 dtype=c['case_snapshot']['declared_tensors']['q'][1]
 if s['operator'].startswith('MLA'):tol={'atol':2**(-7 if dtype=='bfloat16' else -10),'rtol':2**(-7 if dtype=='bfloat16' else -10)}
 elif dtype=='bfloat16':tol={'atol':.016,'rtol':.016}
 else:tol={'policy':'Exact original DSA verification: atol=3e-4,rtol=1e-5; if sentinel index>=S_kv exists, atol=rtol=2e-3. Dynamic predicate executed by unchanged checker/oracle; no fabricated resolved tolerance.'}
 rows.append({'key':key,'accept_recommended':not issues,'issues':issues,'checker':ref(D+s['checker_receipt']),'guard':dict(ref(D+s['guard_receipt']),returncode=g['returncode'],log=guardlog,seconds_after_checker_finished=before),'formal_module':formal,'formal_factory':s['factory'],'tested_registered_identity':{'tested':ref(tested[0][0]),'registered':formal,'byte_identical':True},'cpu_exact_route':route,'seed':seed,'canonical':{'manifest':ref('TileOPs/src/tileops/manifest/spec/attention.yaml'),'workload':wl,'full_declared_shapes_dtypes_parameters_match':True,'snapshot':c['case_snapshot'],'tolerance':tol},'verification':c['verification'],'inputs_preserved':True,'oracle_unchanged':True,'oracle_sources':oracle,'source_provenance':{'candidate_sources':sources,'critical_hashes':str(A/'CRITICAL_SOURCE_HASHES.json'),'native_launch':launch,'native_source_ownership':ownership[:4],'runtime':I['runtime']},'completion':{'checker_source':ref('TileOPs/scripts/ascend_pair_check.py'),'checker_finally_lines':[405,418,440,442],'cleanup_source':ref('reports/baseline-integration-20261026/transformer_attention/ascend_graph_call.py'),'explicit_sync_and_cleanup_before_receipt':True,'guard_running':ref(D+'runs/'+s['case_id']+'/started.json'),'timeout_note':'Full checker, explicit sync and GraphCall cleanup completed before guard124; Python/TBE teardown not clean; guard124 preserved.' if g['returncode']==124 else 'Clean guard exit 0'}})
(A/'RECOMMENDED_ACCEPT.json').write_text(json.dumps({'rows':rows,'NPU_commands':0,'numerical_reruns':0,'registered_shared_files_modified':False},indent=2));print([(r['key'][2],r['accept_recommended'],r['issues']) for r in rows])
# Additional key is separate and never added to scoped recommendations.
c=json.loads(E['files']['reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp8-1024-7168-bfloat16/checker.json']['text']);ch=E['gdn_changes'];assert len(ch)==3;assert all(v['equal'] for v in F['gdn_frozen_equivalence']);assert all(v['expected']==v['sha256'] for v in E['gdn_critical'])
regpath=W+'reports/baseline-integration-700-20261026/transformer_linear/REGISTERED.json';old=F['files'][regpath]['text'];keys=json.loads(old);cid=c['case_id'];new=json.dumps(sorted(set(keys)|{cid}),indent=2)+'\n';patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=regpath,tofile=regpath));(A/'GDN_EXACT_REGISTRATION.patch').write_text(patch)
gdn={'key':['ops-transformer','GDNFwdOp',cid],'decision':'SOURCE_EQUIVALENT_REGISTRATION_PATCH_ONLY','registered_by_auditor':False,'checker':ref('reports/baseline-integration-700-20261026/transformer_linear/'+cid+'/checker.json'),'guard':ref('reports/baseline-integration-700-20261026/transformer_linear/'+cid+'/GUARD_RECEIPT.json'),'verification':c['verification'],'inputs_preserved':c['inputs_preserved'],'oracle_source_changed':c['oracle_source_changed'],'source_change':ch,'source_equivalence':'Provider before/after diff adds only GQADenseFwdOp and MHADecodePagedWithKVCacheFwdOp routes plus their modules; GDN provider binding is unchanged. Exact hash-matched snapshots retained in EXTRA.json.','provider_diff':str(A/'GDN_PROVIDER_CHANGE.diff'),'critical_tested_hash_matches':[{k:v for k,v in t.items() if k!='text'} for t in E['gdn_critical']],'frozen_equivalence':F['gdn_frozen_equivalence'],'current_cpu_route':F['gdn_current_route'],'proposed_factory':'benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_linear_gdn.build','registration_file':ref(regpath),'registration_patch':str(A/'GDN_EXACT_REGISTRATION.patch'),'note':'Patch only proposes adding exact case id to existing REGISTERED.json allowlist. No numerical rerun required for unrelated provider source additions; preserve original source_changed checker status.'}
(A/'GDN_ADDITIONAL_REVIEW.json').write_text(json.dumps(gdn,indent=2));(A/'REMOTE_INDEX.json').write_text(json.dumps(I,indent=2))
