import pathlib,json,hashlib,datetime,collections
P=pathlib.Path(__file__).parent;W=pathlib.Path('/root/dyq/workspace');R='/opt/tileops-all-pairs/workspace';read=lambda f:json.loads((P/f).read_text());h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=read('EVIDENCE.json');n=read('NATIVE_AUDIT.json');c=read('CANONICAL.json');reg=read('REGISTRATION.json');cpu=read('DEVICE_FREE_CHECKS.json')
for path,digest in c['spec_hashes'].items():assert n['files'][R+'/'+path]==digest
pristine=[]
source=e['source_directory']
for path,d in e['hashes'].items():
 if path.startswith(source+'/') and not path.endswith('.so'):
  original=W/'ascend_baselines/sgl-kernel-npu'/path[len(source)+1:]
  assert original.is_file() and h(original)==d['actual'],str(original)
  pristine.append({'upstream':str(original),'execution_source':path,'sha256':h(original)})
assert len(n['batch_progress']['done'])==35 and all(x['status']=='verified' for x in n['batch_progress']['done'])
assert len(cpu['rows'])==len(e['rows'])==reg['registered_count']==36
assert n['native_symbols'] and reg['preserved_prefix']
(P/'ORIGINAL_SOURCE_MATCH.json').write_text(json.dumps(pristine,indent=2))
rows=[]
for r,route in zip(e['rows'],cpu['rows']):
 assert r['key']==route['key']
 rows.append({'baseline':r['key'][0],'operator':r['key'][1],'case_id':r['key'][2],'status':'recommended_accept','registered':True,'factory':route['factory'],'module_sha256':route['module_sha256'],'checker_receipt':r['checker'],'guard_receipt':r['guard'],'canonical':r['canonical'],'seed':r['seed'],'verification':r['verification'],'oracle_source_sha256':r['oracle_source_sha256'],'source_sha256_evidence':'EVIDENCE.json','formal_cpu_route':'DEVICE_FREE_CHECKS.json','all_outputs_and_inputs_preserved':True,'native_library':e['shared_library'],'native_library_sha256':e['hashes'][e['shared_library']]['actual']})
(P/'RECOMMENDED_ACCEPT.json').write_text(json.dumps({'count':36,'parent_review_required':True,'ACCEPTED_modified':False,'rows':rows},indent=2))
delivery={'worker':'sgl_linalg_register36','host':'server2','finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_count':36,'registered':36,'recommended_accept':36,'blocked':0,'counts_by_operator':dict(collections.Counter(r['operator'] for r in rows)),'numerical_reruns':0,'NPU_initializations':0,'torch_npu_imports':0,'device_library_loads':0,'changed_shared_files':[reg['module'],reg['provider']],'local_write_scope':str(P.resolve()),'registration':'REGISTRATION.json','recommendations':'RECOMMENDED_ACCEPT.json','cpu_checks':'DEVICE_FREE_CHECKS.json','full_receipts_and_canonical':'EVIDENCE.json','native_binding_and_dependencies':'NATIVE_AUDIT.json','original_source_match':'ORIGINAL_SOURCE_MATCH.json','canonical_manifest':'CANONICAL.json','guard_summary':'35 batch04 plus 1 cpp03, true guard exits 0; all 36 checker verified 1/1 outputs and inputs preserved; checker finally and adapter explicitly synchronize. Prior failed probes excluded.','implementation_review':'Original full-K public sgemmv_shrink; independent output columns of at most64, zero-aligned padding, layout/group packing and output cast only. Native internal K tiling unchanged. No split-K recombination, schedule/compiler/kernel change or system fallback.','runtime_recipe':{'interpreter':e['interpreter'],'launcher':R+'/reports/baseline-integration-700-20261026/sgl_linalg36/baseline-run','shared_library':e['shared_library'],'note':'Same original runtime overlay and LD_LIBRARY_PATH required; CPU stubs validate route ownership, not numerical behavior.'},'migration':'Module copied byte-for-byte under unique prefix. Absolute SOURCE unchanged; __file__ only changes provenance map key. Formal factory build called for all36 through real registry.','remaining_issues':['Parent must independently review and update ACCEPTED; this worker did not write the acceptance ledger.'],'limitations':['No fresh numerical or performance claim; original successful guarded receipts reused with identical adapter and native hashes.','CPU test intentionally stubs torch, SGL package and data carriers; native ownership additionally verified with nm and original source hashes.']}
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2));print(json.dumps({k:delivery[k] for k in ['registered','recommended_accept','blocked','counts_by_operator']}))
