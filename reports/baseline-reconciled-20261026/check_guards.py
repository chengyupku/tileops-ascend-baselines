import json,sys,pathlib,hashlib
items=json.load(sys.stdin);out=[]
for x in items:
 p=pathlib.Path(x['checker']);s=x['summary'];c=json.load(open(p));guards=[]
 row=x.get('row_summary',{});v=row.get('guard_receipt');explicit=v.get('path') if isinstance(v,dict) else v
 paths=[pathlib.Path(explicit)] if explicit else []
 for parent in [p.parent,p.parent.parent]:
  paths += [f for f in parent.glob('*') if f.is_file() and 'guard' in f.name.lower() and (f.suffix=='.json' or f.name.endswith('guard_exit'))]
 for f in dict.fromkeys(paths):
  try:g=json.load(open(f))
  except:continue
  if isinstance(g,dict):rc=g.get('returncode',g.get('exit_code',g.get('guard_exit_code',g.get('guard_exit'))))
  else:rc=g if type(g)==int else None
  guards.append({'path':str(f),'rc':rc,'data':g if isinstance(g,dict) and len(str(g))<1800 else None})
 out.append({'key':x['key'],'checker':str(p),'checker_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'summary':s,'guards':guards,'result':x['path'],'source_hash_count':len(c.get('source_sha256',{})),'limitations':c.get('limitations'),'adaptation':c.get('adaptation')})
print(json.dumps(out,ensure_ascii=False))
