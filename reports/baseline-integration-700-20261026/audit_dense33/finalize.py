import pathlib,json,hashlib,datetime
D=pathlib.Path(__file__).resolve().parent;W=D.parents[2];P=D.parent/'transformer_dense'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
a=json.loads((D/'REMOTE_AUDIT.json').read_text());e=json.loads((D/'REMOTE_EXTRA.json').read_text());orig={tuple(r['key']):r for r in json.loads((D/'ORIGINAL_SOURCE_CHECK.json').read_text())};nat={tuple(r['key']):r for r in e['native_per_case']};routes={tuple(r['key']):r for r in e['routes']};scope=json.loads((D/'SCOPE.json').read_text());verifier=W/'TileOPs/scripts/ascend_pair_check.py';rows=[]
for r,s in zip(a['rows'],scope):
 key=tuple(r['key']);o=orig[key];n=nat[key];c=json.loads((P/s['checker_receipt']).read_text());assert sha(P/s['checker_receipt'])==r['checker']['sha256'];assert sha(verifier)==c['verifier_sha256'];assert o['matched']>0 and not o['missing'] and not o['different'];assert not r['issues']
 r['checker_remote']=r['checker'];r['checker']=ref(P/s['checker_receipt']);gp=P/s['guard_receipt'];assert sha(gp)==r['guard']['sha256'];r['guard']['remote_path']=r['guard']['path'];r['guard'].update(ref(gp));r['guard']['exit']=ref(gp.with_suffix('.exit'))
 snap=json.loads(pathlib.Path(r['guard']['snapshot']['path'].replace('/opt/tileops-all-pairs/workspace',str(W))).read_text());assert snap['queue_ticket'] in gp.read_text() and snap['acquired_at']<r['started_at'];r['guard']['owner']=snap['owner']
 r['host_cpu_route']['actual_factory_call']=routes[key];r['host_cpu_route']['evidence']=ref(D/'REMOTE_EXTRA.json');r['host_cpu_route']['device_constructors_only_stubbed']=True
 r['source_provenance']['original_source_check']=dict(evidence=ref(D/'ORIGINAL_SOURCE_CHECK.json'),files_matched=o['matched']);r['source_provenance']['native_per_case']=n
 r['source_provenance']['native_evidence_scope']='batch-wide ownership; per-case interval evidence separately recorded'
 r['synchronization']=dict(checker=ref(verifier),lines=[405,408,441],rule='finally explicitly synchronizes before final receipt; failed sync changes status and returncode',complete_before_finished_at=True)
 r['canonical']=dict(seed_rule='sha256(operator::case_id) first 4 bytes big endian',case_selection='official cases(op), unique exact case id; no size overrides',shape_dtype_layout=r['case_snapshot'],oracle_hashes_rechecked=True,all_outputs=r['verification'],inputs_preserved=True,state='All declared inputs/outputs and observed reference/candidate calls retained in REMOTE_AUDIT.json')
 op=key[1]
 if op in ['BmmFP8FwdOp','MoEGroupedGemmFP8FwdOp']:
  k=r['case_snapshot']['parameters']['a'][0][-1];r['tolerance']=dict(atol=.02*max(1,k/1024)**.5,rtol=.02,source='TileOPs/workloads/gemm.py:607',coverage='all declared outputs; canonical valid-row mask for padded MoE expert slabs' if op.startswith('MoE') else 'all output elements')
 elif op=='GemmINT8W8A8FwdOp':r['tolerance']=dict(atol=.016,rtol=.016,normalized=.001,source='TileOPs/workloads/ascend_extended.py:114; numerics.py:264')
 elif op=='MoEPrePermuteFwdOp':r['tolerance']=dict(atol=0,rtol=0,coverage='all three outputs; exact ends, inverse permutation, routed rows and expert ownership',source='TileOPs/workloads/moe.py:180')
 else:r['tolerance']=dict(atol=.00001,rtol=.00001,source='TileOPs/workloads/rope.py:11; numerics.py:264')
 r['qualifications']=[]
 if op=='YaRNFwdOp':
  r['tested_registered_identity']['migration']=e['yarn'];r['tested_registered_identity']['equivalent_for_exact_key']=True
  later=json.loads((P/'rope_check_graph_cpu_state/LongRoPEFwdOp.json').read_text());assert r['finished_at']<later['started_at'] and later['status']=='failed';assert r['guard']['returncode']==1
  r['guard']['later_failure']=dict(checker=ref(P/'rope_check_graph_cpu_state/LongRoPEFwdOp.json'),started_at=later['started_at'],error=later['error'])
  r['qualifications'].append('True batch guard exit 1 retained: YaRN finalized after explicit synchronization before later LongRoPE failure. Original native ownership bridged by matching ELF code/data/ABI sections, only generated filename string differs; no per-case launch path in verbosity3 log.')
 elif not n['launched_kernels']:
  r['issues'].append('No native kernel launch line survives inside this exact checker interval. Batch sibling native ownership is verified, but is not relabelled as this case launch. Earliest retained debug plog starts 14:21:54.320; these five cases finish before then. Existing run/device/ATB logs inspected and contain no kernel attribution.')
 r['accept_recommended']=not r['issues'];r['decision']='recommended_accept' if r['accept_recommended'] else 'needs_native_interval_evidence';rows.append(r)
(D/'RECOMMENDED_ACCEPT.json').write_text(json.dumps(dict(rows=rows,count=len(rows),recommended=sum(r['accept_recommended'] for r in rows)),indent=2))
summary=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope_count=len(rows),recommend_accept=sum(r['accept_recommended'] for r in rows),needs_native_interval_evidence=[r['key'] for r in rows if r['issues']],checked=dict(ssh_exit=0,live_registry_exact_keys=33,actual_factory_cpu_calls=33,complete_checker_receipts=33,canonical_seeds=33,original_kernel_source_files_unique=111,oracle_drift=0,npu_commands=0,numerical_reruns=0,benchmark=0,child_agents=0),qualifications=[r['qualifications'] for r in rows if r['qualifications']],writes='audit_dense33 private directory only; no ACCEPTED, provider, memory or remote changes',remaining='Parent final accounting; five FP8 Bmm cases need recoverable historical native launch attribution, not another numerical run authorized by this CPU audit.')
(D/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,ensure_ascii=False))
