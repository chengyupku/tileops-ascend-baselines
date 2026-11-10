from pathlib import Path
import sys,json,fcntl,hashlib,shutil,ast,datetime
p=Path(__file__).parent;d=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer');batch=sys.argv[1];assert batch.isalnum()
mods={'GemmINT8W8A8FwdOp':('quant','quant'),'BmmFP8FwdOp':('fp8_bmm','fp8_bmm'),'MoEGroupedGemmFP8FwdOp':('fp8_moe','fp8_moe')};scope={(x['baseline'],x['operator'],x['case_id']) for x in json.loads((p/'SCOPE.json').read_text())};done=set()
for f in p.glob('REGISTRATION_SIBLINGS*.json'):
 done.update(tuple(x['key']) for x in json.loads(f.read_text())['rows'])
rows=[];byop={}
for f in [*(p/'all_siblings').glob('*.json'),*p.glob('fp8_siblings*/*.json')]:
 c=json.loads(f.read_text());key=(c.get('library'),c.get('op'),c.get('case_id'))
 if c.get('status')!='verified' or key in done:continue
 assert key in scope and key[1] in mods
 v=c['verification'];assert v['checked_outputs']==v['total_outputs'] and v['total_outputs']>0
 mod,func=mods[key[1]];name=f'integration700_transformer_dense_{mod}.py';sha=hashlib.sha256((p/name).read_bytes()).hexdigest();tested=[h for n,h in c['source_sha256'].items() if n.endswith('/'+name)];assert tested==[sha]
 byop.setdefault(key[1],[]).append(key[2]);rows.append(dict(key=key,factory=f'benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_dense_{mod}.{func}',module_sha256=sha,checker_receipt=str(f.relative_to(p)),registered=True))
assert rows,'No new verified keys'
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);src=(d/'__init__.py').read_text();backup=f'provider_init.siblings{batch}.before.py';assert not (p/backup).exists();shutil.copy2(d/'__init__.py',p/backup);block=f'\n# integration700_transformer_dense verified siblings {batch}\n'
 for op,ids in byop.items():
  mod,func=mods[op];name=f'integration700_transformer_dense_{mod}.py';assert (d/name).read_bytes()==(p/name).read_bytes();tag=f'_integration700_transformer_dense_{batch}_{op}';assert tag not in src
  block+=f'from .integration700_transformer_dense_{mod} import {func} as {tag}_factory\n{tag}_old = PROVIDERS.get({op!r})\ndef {tag}_dispatch(case):\n    if case.id in {ids!r}: return {tag}_factory(case)\n    if {tag}_old is not None: return {tag}_old(case)\n    from benchmarks.ascend_baselines import UnsupportedCase\n    raise UnsupportedCase("No exact integration700 key")\nPROVIDERS[{op!r}] = {tag}_dispatch\n'
 new=src+block;ast.parse(new);tmp=d/'integration700_transformer_dense_init.tmp';tmp.write_text(new);tmp.replace(d/'__init__.py');(p/f'provider_init.siblings{batch}.patch').write_text(block);(p/f'REGISTRATION_SIBLINGS{batch}.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,lock='/tmp/baseline700-provider-ops_transformer.lock',init_before_sha256=hashlib.sha256(src.encode()).hexdigest(),init_after_sha256=hashlib.sha256(new.encode()).hexdigest()),indent=2))
print('registered',len(rows),'exact keys')
