import pathlib,json,hashlib,subprocess,datetime,fcntl
D=pathlib.Path(__file__).resolve().parent;A=D/'acceptance_audit01';rows=[r for r in json.loads((A/'DENSE_RECOMMENDED_ACCEPT.json').read_text())['rows'] if r['accept_recommended']];assert len(rows)==4
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for r in rows:
 for typ in ['checker','guard','actual_guard_exit','guard_running']:assert sha(r[typ]['path'])==r[typ]['sha256']
 c=json.loads(pathlib.Path(r['checker']['path']).read_text());v=c['verification'];assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'];assert v['checked_outputs']==v['total_outputs']>0;assert r['actual_guard_exit']['returncode']==0 and pathlib.Path(r['actual_guard_exit']['path']).read_text().strip()=='0'
 g=json.loads(pathlib.Path(r['guard_running']['path']).read_text());assert g['owner']=='baseline700-20261026/transformer_dense';assert g['acquired_at']<c['started_at'];log=pathlib.Path(r['guard']['path']).read_text();assert g['queue_ticket'] in log;assert not any(x in log.lower() for x in ['interference detected','external npu process','terminating process group'])
 for x in r['source_provenance']['native_evidence']:
  if 'path' in x:assert sha(x['path'])==x['sha256']
proof=json.loads((A/'TOPK_IMPORT_RELOCATION_PROOF.json').read_text());topk=(D/'transformer_dense/integration700_transformer_dense_topk.py').read_text();old=topk.replace('    try:from .integration700_transformer_dense_graph import GraphCall\n    except ImportError:from integration700_transformer_dense_graph import GraphCall\n','    from integration700_transformer_dense_graph import GraphCall\n');assert hashlib.sha256(old.encode()).hexdigest()==proof['tested_sha256']
code='''import sys,importlib,types,pathlib,hashlib,fcntl,json
sys.dont_write_bytecode=True
sys.path.insert(0,'/opt/tileops-all-pairs/workspace/TileOPs')
rows=ROWS
out=[]
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 p=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
 for r in rows:
  mod,fn=r['formal_factory'].rsplit('.',1);m=importlib.import_module(mod);h=hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest();assert h==r['module_sha256'];old=getattr(m,fn);stub=lambda *a: ('expected',r['key']);aliases={k:v for k,v in vars(p).items() if v is old};setattr(m,fn,stub)
  for k in aliases:setattr(p,k,stub)
  case=types.SimpleNamespace(id=r['key'][2],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['key'][1])))
  assert p.PROVIDERS[r['key'][1]](case)==('expected',r['key'])
  setattr(m,fn,old)
  for k,v in aliases.items():setattr(p,k,v)
  out.append(dict(key=r['key'],module_sha256=h,exact_route=True))
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 print(json.dumps(dict(rows=out,device_imported=False)))
'''.replace('rows=ROWS','rows='+repr(rows))
S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';r=subprocess.run([S,'openlibing-DevEnv-661495','python3 -S -'],input=code,text=True,capture_output=True,timeout=50);assert r.returncode==0,r.stderr;(D/'PARENT_DENSE4_ROUTE_CHECK.json').write_text(r.stdout)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();out=D/'ACCEPTED.json'
with open(str(out)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads(out.read_text());by={tuple(x['key']):x for x in d['rows']};before=len(by)
 for x in rows:by[tuple(x['key'])]=dict(x,at=now,accepted=True,parent_review=str(D/'PARENT_DENSE4_REVIEW.json'))
 d.update(rows=list(by.values()),updated_at=now);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False));tmp.replace(out)
review=dict(at=now,accepted_added=len(by)-before,accepted_total=len(by),keys=[x['key'] for x in rows],evidence=str(A/'DENSE_RECOMMENDED_ACCEPT.json'),parent_checks=['Original fullMoeGatingTopK,RotaryPositionEmbedding,BiasGateGelu source bindings reviewed.','Rechecked every checker/guard/guardexit/metadata and native log file digest, fulloutputs/inputpreservation/oracle unchanged.','TopK tested source reconstructed digest equals receipt; formal change only relative import to identical GraphCall.','Live formal dispatch and modulehash checked4cases onserver1CPUonly.','YaRN excluded until original kernel launch and tested/live source identity supplied.'],audit_worker_wall_s=804.625688,audit_NPU_commands=0,audit_numerical_reruns=0)
(D/'PARENT_DENSE4_REVIEW.json').write_text(json.dumps(review,indent=2));print(json.dumps(review))
