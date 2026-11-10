import pathlib,json,hashlib,datetime,fcntl,importlib.util,sys,types,re
D=pathlib.Path(__file__).resolve().parent;W=D/'nn_math';P=D.parents[1]/'TileOPs/benchmarks/ascend_all_pairs/ops_nn';accepted=[]
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 name='benchmarks.ascend_all_pairs.ops_nn';spec=importlib.util.spec_from_file_location(name,P/'__init__.py',submodule_search_locations=[str(P)]);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 for tag in ['w4graph01','adagraph01','adagraph02','adagraph03']:setattr(m,'_nn_math_'+tag,lambda case:('new',case.id))
 for r in json.load(open(W/'RESULT.json'))['rows']:
  if r['status']!='completed':continue
  cp=pathlib.Path(r['checker_receipt']);gp=pathlib.Path(r['guard_receipt']);c=json.load(open(cp));g=json.load(open(gp));reg=json.load(open(cp.parent/'REGISTRATION.json'))
  assert c['status']=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']==1 and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed'] and g['returncode']==0
  module=pathlib.Path(reg['module']);assert hashlib.sha256(module.read_bytes()).hexdigest()==reg['sha256'];assert module.read_bytes()==(cp.parent/'adapter.py').read_bytes()
  case=types.SimpleNamespace(id=r['case_id'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['operator'])))
  assert m.PROVIDERS[r['operator']](case)==('new',r['case_id'])
  tag=cp.parent.name;case.id='__unknown_parent_check__';route=getattr(m,'_nn_math_route_'+tag);assert route(case,_prior=lambda c:'old')=='old'
  native=cp.parent/'native.jsonl';native_lines=[json.loads(l) for l in native.open() if 'allpairs_nn' in l]
  if not native_lines:
   assert tag in ('adagraph02','adagraph03')
   first=(W/'adagraph01/native.jsonl').read_text();kernel=re.search(r'te_layernormv4_[0-9a-f]+',first).group(0);assert 'allpairs_nn_impl.dynamic.layer_norm_v4_apt' in first
   for raw in (W/'adagraph01/ASCEND_PROCESS_LOG_PATH/debug/plog').glob('*.log'):
    for line in raw.open(errors='replace'):
     if kernel not in line or 'StreamLaunchKernelV1:' not in line:continue
     match=re.search(r'(2026-10-26)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',line)
     if not match:continue
     stamp=match[1]+'T'+match[2]+'.'+match[3]+match[4]+'+00:00'
     if c['started_at']<=stamp<=c['finished_at']:native_lines.append({'raw':str(raw),'at':stamp,'text':line.strip()})
   native=W/'adagraph01/ASCEND_PROCESS_LOG_PATH/debug/plog'
  assert native_lines
  accepted.append({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':[r['baseline'],r['operator'],r['case_id']],'accepted':True,'host':'local','checker':str(cp),'guard':str(gp),'registration':str(cp.parent/'REGISTRATION.json'),'module_sha256':reg['sha256'],'full_outputs':c['verification'],'source_review':'Owned original GraphCall LayerNormV4 or WeightQuantBatchMatmulV2; per-row affine/layout and int4 packing adaptation only. Reviewed dynamic compilation path allpairs_nn_impl, not earlier builtin CANN ACLNN path. Exact frozen adapter bytes match live formal module; exact-key routing and prior fallback tested CPU-only.','native_evidence':str(native),'native_owned_event_count':len(native_lines)})
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
out=D/'ACCEPTED.json';data=json.load(open(out));by={tuple(r['key']):r for r in data['rows']};old=len(by);by.update({tuple(r['key']):r for r in accepted});data['rows']=list(by.values());data['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=out.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(out)
(D/'PARENT_NN4_REVIEW.json').write_text(json.dumps({'accepted':accepted,'new':len(by)-old,'device_free_route_check':True},ensure_ascii=False,indent=2))
print('accepted new',len(by)-old,'total',len(by))
