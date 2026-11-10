import ast,json,pathlib,hashlib,textwrap
r=pathlib.Path(__file__).resolve().parent
old=(r/'math_bindings.before-v31.py').read_text();new=(r/'math_bindings.current.py').read_text();expected='532000173a31623caa7ae7510ba4b9b775a85ac169bb57b55f5852d10ce8e22a'
assert hashlib.sha256(old.encode()).hexdigest()==expected
assert new.startswith(old)
added=ast.parse(textwrap.dedent(new[len(old):]));assert len(added.body)==2 and isinstance(added.body[0],ast.Import) and isinstance(added.body[1],ast.If)
assert ast.dump(added)==ast.dump(ast.parse("import os\nif os.environ.get('RETEST32_ORIGINAL_VECTOR_NORM')=='v31':\n from tools.baseline_retest32_original_composed_v10 import make as original_make\n NATIVE_CANDIDATES['VectorNormFwdOp']=partial(original_make,op='VectorNormFwdOp')\n"))
p=r/'server2-baseline-retest32-20261030-audit.json';d=json.loads(p.read_text());affected=[x for x in d['rows'] if x['errors']];assert len(affected)==71
for x in affected:
 assert x['errors']==['changed source/component: tools/baseline_retest32_math_bindings.py'];assert x['key'][1]!='VectorNormFwdOp'
 x['resolved_errors']=x['errors'];x['errors']=[];x['notes'].append('Exact historical source retained and SHA verified; later appended conditional only adds unrelated VectorNorm dispatch. Original source prefix byte-identical. Admits frozen result, not reuse with changed live source.')
d['usable']=len(d['rows']);d['errors']={};d['historical_source_resolution']={'expected_sha256':expected,'archive':'math_bindings.before-v31.py','current':'math_bindings.current.py','affected':71,'change':'Appended VectorNorm-only conditional; none of the affected rows uses VectorNorm; original prefix byte-identical.'}
(r/'server2-baseline-retest32-20261030-audit-resolved.json').write_text(json.dumps(d,indent=2)+'\n')
c=json.loads((r/'server2-catalog.json').read_text());ids={x['cache_key'] for x in affected}
for x in c:
 if x['cache_key'] in ids:x['historical_source_resolution']=d['historical_source_resolution'];x['errors']=[]
(r/'server2-catalog-resolved.json').write_text(json.dumps(c)+'\n');print('server2 audited',len(c),'rows; errors',sum(bool(x['errors']) for x in c))
