"""Archive only immutable completed attempts, retaining full raw profiles."""
import pathlib,json,hashlib,tarfile
P=pathlib.Path(__file__).resolve().parent;D=P/'evidence';D.mkdir(exist_ok=True)
indexfile=D/'INDEX.json';index=json.loads(indexfile.read_text()) if indexfile.exists() else {}
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
for measurement in sorted((P/'runs').glob('*/*/measurement.json')):
 out=measurement.parent;key=str(out.relative_to(P));mh=sha(measurement)
 if key in index:
  assert index[key]['measurement_sha256']==mh
  continue
 valid=[]
 for f in out.glob('guard*.json'):
  try:g=json.loads(f.read_text())
  except json.JSONDecodeError:continue
  if g.get('phase')=='finished' and g.get('performance_valid') and g.get('workload_exit_code')==0:valid.append(f)
 if not valid:continue
 archive=D/(out.parent.name+'-'+out.name+'.tar.gz')
 with tarfile.open(archive,'w:gz',compresslevel=1,dereference=True) as tar:
  for f in sorted(out.rglob('*')):
   if not f.is_file() or f.name=='payload.json':continue
   tar.add(f,arcname=str(f.relative_to(P)),recursive=False)
  m=json.loads(measurement.read_text());candidate=m['source_sha256']['candidate_file']
  tasks=[f for f in (P/'tasks').glob('*.json') if json.loads(f.read_text()).get('id')==m['task_id'] and json.loads(f.read_text()).get('candidate_file')==candidate]
  assert len(tasks)==1,(out,tasks)
  tar.add(tasks[0],arcname='task.json')
  for glob in ['worker*.py','candidate*.py','launch*.sh']:
   for f in P.glob(glob):tar.add(f,arcname='implementation/'+f.name)
  for name in ['OLD_SOURCE.json','PRECISION_POLICY.json','ENVIRONMENT_EQUIVALENCE.json']:
   if (P/name).exists():tar.add(P/name,arcname=name)
 index[key]={'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'measurement_sha256':mh,'guard_sha256':sha(valid[-1])}
 indexfile.write_text(json.dumps(index,indent=2));print(key,archive.stat().st_size,flush=True)
print('Completed evidence archives:',len(index),flush=True)
