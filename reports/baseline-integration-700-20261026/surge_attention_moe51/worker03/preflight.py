import ast, datetime, hashlib, json, pathlib, re, time
start=time.monotonic()
root=pathlib.Path('/root/dyq/workspace')
out=pathlib.Path(__file__).resolve().parent
scope=json.loads((out/'SCOPE.json').read_text())
prior_path=root/'reports/baseline-integration-700-20261026/threehost_attention_moe53/CURRENT_SOURCE_AUDIT.json'
prior=json.loads(prior_path.read_text())
def capture(p):
    b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest()}
evidence={'prior_audit':capture(prior_path),'api':{},'manifest':capture(root/'TileOPs/src/tileops/manifest/spec/attention.yaml'),'op_contract':capture(root/'TileOPs/src/tileops/ops/attention/gqa/paged.py')}
for name in ['flash_attn','quant_flash_attn']:
    p=root/f'ascend_baselines/ops-transformer/attention/{name}/torch_extension/{name}.py'
    tree=ast.parse(p.read_text())
    funcs=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    assert funcs, name
    fn=funcs[0]
    info=capture(p)
    info['function_line']=fn.lineno
    info['parameters']=[a.arg for a in fn.args.args+fn.args.kwonlyargs]
    assert not any(re.search('rope|rotary|softcap|soft_cap',a,re.I) for a in info['parameters'])
    host=root/f'ascend_baselines/ops-transformer/attention/{name}/op_host/{name}_def.cpp'
    info['host_schema']=capture(host)
    info['host_schema']['inputs_and_attrs']=re.findall(r'(?:Input|Attr)\("([^"]+)"\)',host.read_text())
    assert not any(re.search('rope|rotary|softcap|soft_cap',a,re.I) for a in info['host_schema']['inputs_and_attrs'])
    evidence['api'][name]=info
assert evidence['api']['flash_attn']['sha256']==prior['flash_api']['sha256'], 'Original API changed; inspect delta before concluding'
evidence['flash_api_unchanged_from_prior']=True
lines=(root/'TileOPs/src/tileops/manifest/spec/attention.yaml').read_text().splitlines()
rows=[]
for idx,s in enumerate(scope,1):
    cid=s['key'][2]
    label=re.sub(r'-(float16|bfloat16)-(float16|bfloat16)$','',cid)
    matches=[{'line':i+1,'text':line} for i,line in enumerate(lines) if 'label: '+label in line and 'NP:' in line]
    assert len(matches)==1,(cid,matches)
    text=matches[0]['text']
    soft='softcap' in cid
    assert ('softcap: 50.0' if soft else 'pos_encoding_mode: rope') in text
    r={'key':s['key'],'task_type':'实际对接/补测','status':'blocked_original_api_contract','numerical_status':'not_run','executed_this_task':False,'registered':False,'runtime_ready':False,'accepted_added':0,'checked_outputs':0,'total_outputs':1,'prior_scope_path':s['prior_scope_path'],'prior_status':s['status'],'prior_reason':s['reason'],'reason':('Exact case requires score softcap=50; original FlashAttn and QuantFlashAttn schemas have no softcap attribute. Adding tanh score computation is outside authorized thin adaptation.' if soft else 'Exact case requires internal RoPE of Q and cached K; original FlashAttn and QuantFlashAttn schemas lack rotary table/position inputs. External rotation is outside authorized thin adaptation.'),'manifest_evidence':matches[0],'source_evidence':'EVIDENCE.json','reused_audit':str(prior_path)+'#flash_api','changed_retry_condition':None,'device_attempts':[],'guard_receipt':None,'fullchecker_receipt':None,'native_evidence':None,'seed':None,'seed_note':'No checker executed; no seed instantiated or changed.','timings':{k:None for k in ['requested_utc','acquired_utc','child_start_utc','end_utc','guard_exit','queue_seconds','physical_idle_wait_seconds','holding_lock_cpu_seconds','device_seconds']},'next_step':'Needs original API support; no allowed parameter-only repair exists in inspected public schema. Do not rerun unchanged.'}
    (out/f'RESULT_{idx:02d}.json').write_text(json.dumps(r,indent=2)+'\n')
    rows.append(r)
(out/'EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owner':'baseline700-20261026/surge_attention_moe51/worker03','results':rows,'success_count':0,'accepted_added':0,'blocked_count':7,'device_attempts':0,'three_host_pre_submission_check':'Not triggered: no device submission; host/ABI changes cannot supply missing mathematical API inputs. No external-occupancy claim made.','preflight_script_seconds':time.monotonic()-start,'command':'python3 '+str(out/'preflight.py'),'checks':['7 unique scope keys','exact paged manifest entry per case','AST original API parameters','native host schema inputs/attributes','original FlashAttn SHA256 matches reused audit'],'remaining_keys':[s['key'] for s in scope]}
assert len({tuple(s['key']) for s in scope})==7
(out/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'HEARTBEAT.json').write_text(json.dumps({'utc':summary['utc'],'exact_keys':summary['remaining_keys'],'phase':'completed_blocked','last_output':'7 per-key RESULT files, EVIDENCE.json, RESULT.json','blocker':'Original API lacks required softcap/RoPE; no authorized thin repair','next':'Coordinator review; no unchanged NPU replay'},indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['success_count','accepted_added','blocked_count','device_attempts','checks']}))
