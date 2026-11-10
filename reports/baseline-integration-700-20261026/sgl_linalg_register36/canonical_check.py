import json,pathlib,hashlib,yaml
P=pathlib.Path(__file__).parent;W=pathlib.Path('/root/dyq/workspace');e=json.loads((P/'EVIDENCE.json').read_text());scope=json.loads((P/'SCOPE.json').read_text())
assert {tuple(r['key']) for r in e['rows']}=={(r['baseline'],r['operator'],r['case_id']) for r in scope}
spec={};hashes={}
for f in ['gemm.yaml','gemm_bmm.yaml']:
 p=W/'TileOPs/src/tileops/manifest/spec'/f;spec.update(yaml.safe_load(p.read_text()));hashes[str(p.relative_to(W))]=hashlib.sha256(p.read_bytes()).hexdigest()
for r in e['rows']:
 _,op,cid=r['key'];matches=[(x,t) for x in spec[op]['workloads'] for t in x['dtype_cases'] if x['label']+'-'+t['T']==cid];assert len(matches)==1
 x,t=matches[0];M,N,K=x['M'],x['N'],x['K'];dt=t['T'];params=r['case_snapshot']['parameters'];tol=.001 if dt=='float16' else .016;atol=tol
 if op=='BmmFwdOp':shapes={'a':[x['B'],M,K],'b':[x['B'],K,N],'d':[x['B'],M,N]}
 elif op=='GemmFwdOp':
  ta=x.get('trans_a',False);tb=x.get('trans_b',True);shapes={'a':[K,M] if ta else [M,K],'b':[N,K] if tb else [K,N],'d':[M,N]};assert params['trans_a']==ta and params['trans_b']==tb
  if not ta and ((M==1 and tb) or (N==1 and not tb)):atol*=max(1,K/2048)
 else:
  ta=x['transpose_a'];tb=x['transpose_b'];G=x['G'];assert not ta
  shapes={'a':[M,K],'b':[G,N,K] if tb else [G,K,N],'output':[M,N],'batch_sizes':[G],'batch_offsets':[G]}
  assert params['transpose_a']==ta and params['transpose_b']==tb
 for name,shape in shapes.items():
  assert params[name]==[shape,'int32' if name.startswith('batch_') else dt],(cid,name,params)
 assert r['seed']==int.from_bytes(hashlib.sha256(f'{op}::{cid}'.encode()).digest()[:4],'big')
 r['canonical']={'manifest_workload':x,'dtype':t,'rtol':tol,'atol':atol,'all_outputs':True,'state':'No mutable state; checker inputs_preserved=true'}
(P/'EVIDENCE.json').write_text(json.dumps(e,indent=2));(P/'CANONICAL.json').write_text(json.dumps({'count':len(e['rows']),'spec_hashes':hashes},indent=2));print('canonical',len(e['rows']))
