import json,pathlib,hashlib,sys,types,ast
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_linear'
ids=['q35-2b-65536-bfloat16','q35-2b-8x8192-bfloat16','q35-35b-16384-bfloat16','q35-35b-32768-bfloat16','q35-397b-tp1-16384-bfloat16','q35-397b-tp2-32768-bfloat16','q35-397b-tp4-65536-bfloat16','q35-397b-tp4-8x8192-bfloat16','q35-397b-tp8-16x8192-bfloat16']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
mp=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_transformer_linear_gdn.py';m=types.ModuleType('audit');exec(compile(mp.read_text(),str(mp),'exec'),m.__dict__)
# wrap executes real dispatch; build is boundary stub so no runtime initialization.
m.build=lambda c:('formal-build',c.id)
route=m.wrap(lambda c:('prior',c.id));res={'module':{'path':str(mp),'sha256':sha(mp)},'rows':[]}
ip=mp.parent/'__init__.py';res['provider']={'path':str(ip),'sha256':sha(ip),'registration_lines':[x for x in ip.read_text().splitlines() if 'gdn' in x.lower() and ('wrap' in x or 'integration700_transformer_linear' in x or 'PROVIDERS' in x)]}
for cid in ids:
 cp=P/cid/'checker.json';c=json.loads(cp.read_text());bad=[];hashes={}
 chosen={k:v for k,v in c['source_sha256'].items() if '/chunk_package/' in k and k.endswith(('binding.py','direct_acl.py'))}
 chosen.update({str(W/k):v for k,v in c['oracle_source_sha256'].items()})
 for p,h in chosen.items():
  hashes[p]=sha(p)
  if hashes[p]!=h:bad.append(p)
 for fn in ['binding.py','direct_acl.py']:
  live=P/'frozen_v1/chunk_package'/fn;hashes[str(live)]=sha(live)
  if sha(live)!=sha(P/'chunk_package'/fn):bad.append(str(live))
 loaded=json.loads((P/cid/'loaded-libraries.json').read_text());primary=[x for x in loaded if ('recovery0730-gdn-fixed-abi/chunk_binary/' in x['path'] and x['path'].endswith('libcust_opapi.so')) or x['path'].endswith('libnnopbase.so')]
 for x in primary:
  hashes[x['path']]=sha(x['path'])
  if hashes[x['path']]!=x['sha256']:bad.append(x['path'])
 if len(primary)!=2:bad.append('Primary runtime binary missing from loaded map')
 res['rows'].append({'case_id':cid,'checker_sha256':sha(cp),'hashes':hashes,'runtime_primary':primary,'dispatch':route(types.SimpleNamespace(id=cid)),'unknown':route(types.SimpleNamespace(id='__audit_unknown__')),'issues':bad})
print(json.dumps(res))
