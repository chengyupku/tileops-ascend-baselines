import pathlib,json,hashlib,datetime,fcntl,sys,types,re,importlib
sys.dont_write_bytecode=True
D=pathlib.Path(__file__).resolve().parent;W=D.parents[1];N=D/'nn_recovery01';O=D/'nn_parent_last3';O.mkdir(exist_ok=True);frozen=O/'frozen_adapters/adapter_037f902f67.py';frozen.parent.mkdir(exist_ok=True);frozen.write_bytes((N/'gemmw4a16fwdop_1_0/adapter.py').read_bytes());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
previous={tuple(x['key']) for x in json.loads((D/'nn_registration01/REGISTRATION.json').read_text())['rows']};rows=[r for r in json.loads((N/'RESULT.json').read_text())['rows'] if r['status']=='verified_unregistered' and tuple(r[k] for k in ['baseline','operator','case_id']) not in previous];assert len(rows)==3 and all(r['operator']=='GemmW4A16FwdOp' for r in rows)
proof=[]
for r in rows:
 cp=pathlib.Path(r['checker_receipt']);c=json.loads(cp.read_text());gp=pathlib.Path(r['guard_receipt']);g=json.loads(gp.read_text());ap=cp.parent/'ATTRIBUTION.json';a=json.loads(ap.read_text());assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'];assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert g['returncode']==0 and g['requested_at']<c['started_at']<c['finished_at']<g['finished_at'];meta=json.loads((cp.parent/'GUARD_METADATA.json').read_text());assert meta['owner']=='baseline700-20261026/nn_recovery01'
 batch=json.loads(pathlib.Path(json.loads((gp.parent/'COMMAND.json').read_text())[-1]).read_text());assert [r['baseline'],r['operator'],r['case_id'],cp.parent.name] in batch
 assert sha(cp.parent/'adapter.py')==sha(frozen)==c['source_sha256'][str((cp.parent/'adapter.py').relative_to(W))]
 assert a['api']=='WeightQuantBatchMatmulV2' and a['vendor_seen'] and a['registration_seen'] and a['device_launch_seen'];launches=[]
 for x in a['records']:
  if 'StreamLaunchKernelV1:' not in x['text']:continue
  line=pathlib.Path(x['file']).read_text(errors='replace').splitlines()[x['line']-1];assert line.startswith(x['text']);mt=re.search(r'(2026-\d\d-\d\d)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',line);t=mt[1]+'T'+mt[2]+'.'+mt[3]+mt[4]+'+00:00';assert c['started_at']<=t<=c['finished_at'];kernel=re.search(r'kernel_name=([^, ]+)',line)[1];assert any(pathlib.Path(z).stem in kernel for z in a['kernel_hashes']);launches.append(dict(file=x['file'],line=x['line'],at=t,kernel=kernel))
 assert launches
 for f,h in a['kernel_hashes'].items():assert sha(f)==h
 for name in ['graph.py','common.py','catalog.py','truth.py','tensorlist.py']:
  q='TileOPs/benchmarks/ascend_all_pairs/ops_nn/'+name;assert sha(W/q)==c['source_sha256'][q]
 for f,h in c['oracle_source_sha256'].items():assert sha(W/f)==h
 proof.append(dict(key=[r['baseline'],r['operator'],r['case_id']],checker=dict(path=str(cp),sha256=sha(cp)),guard=dict(path=str(gp),sha256=sha(gp),returncode=0),native_attribution=dict(path=str(ap),sha256=sha(ap),launches=launches),frozen_adapter=dict(path=str(frozen),sha256=sha(frozen)),full_outputs=c['verification']))
P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn';mp=P/'integration700_nn_parent_last3.py';ip=P/'__init__.py';keys={(r['operator'],r['case_id']) for r in rows};source=(P/'integration700_nn_registration01.py').read_text().replace(str(D/'nn_registration01/frozen_adapters'),str(frozen.parent)).replace('adapter_68ce198132.py','adapter_037f902f67.py');start=source.index('_KEYS = {');end=source.index('\n\ndef _adapter_for',start);source=source[:start]+'_KEYS = '+repr(keys)+'\n'+source[end:];marker='# Exact registration: baseline700 parent last3 W4\n'
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);before=ip.read_text();assert marker not in before;assert not mp.exists();(O/'provider_init.before.py').write_text(before);mp.write_text(source)
 append=marker+'from . import integration700_nn_parent_last3 as _i700_nn_parent_last3\n_i700_nn_parent_last3_previous = PROVIDERS["GemmW4A16FwdOp"]\ndef _i700_nn_parent_last3_route(case):\n    if ("GemmW4A16FwdOp", case.id) in _i700_nn_parent_last3._KEYS:\n        return _i700_nn_parent_last3.make(case)\n    return _i700_nn_parent_last3_previous(case)\nPROVIDERS["GemmW4A16FwdOp"] = _i700_nn_parent_last3_route\n';ip.write_text(before+'\n'+append)
 sys.path.insert(0,str(W/'TileOPs'));from benchmarks.ascend_all_pairs import ops_nn
 from benchmarks.ascend_all_pairs.ops_nn import integration700_nn_parent_last3 as m
 old=m._adapter_for;m._adapter_for=lambda op:types.SimpleNamespace(make=lambda case:('expected',op,case.id))
 for op,cid in keys:
  case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)));assert ops_nn.PROVIDERS[op](case)==('expected',op,cid)
 prior=ops_nn._i700_nn_parent_last3_previous;ops_nn._i700_nn_parent_last3_previous=lambda case:'prior';case.id='__unknown_parent_route__';assert ops_nn.PROVIDERS['GemmW4A16FwdOp'](case)=='prior';ops_nn._i700_nn_parent_last3_previous=prior;m._adapter_for=old
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 for r in proof:r.update(formal_factory='benchmarks.ascend_all_pairs.ops_nn.integration700_nn_parent_last3.make',module_sha256=sha(mp),exact_route_checked=True,prior_preserved=True)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();(O/'REGISTRATION.json').write_text(json.dumps(dict(at=now,rows=proof,numeric_reruns=0,source_boundary='Exact tested037f902f adapter frozen; W4A16 branch unchanged from acceptedNN18/oldNN4; addedexactkeys only, native source/runtime unchanged.'),indent=2));out=D/'ACCEPTED.json'
with open(str(out)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads(out.read_text());by={tuple(x['key']):x for x in d['rows']}
 for r in proof:assert tuple(r['key']) not in by;by[tuple(r['key'])]=dict(r,accepted=True,at=now,parent_review=str(O/'REGISTRATION.json'))
 d.update(rows=list(by.values()),updated_at=now);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False));tmp.replace(out)
print(json.dumps(dict(added=3,total=len(by),keys=[r['key'] for r in proof],device_runs=0)))
