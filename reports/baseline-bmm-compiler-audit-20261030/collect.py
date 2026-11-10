import os,sys,json,hashlib,subprocess,time,csv,base64
from pathlib import Path
base=Path('/usr/local/Ascend/cann-9.1.0');workspace=Path(sys.argv[1]);files={};errors=[];cache={};links={}
def sha(p):
 s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
def tree(p,label):
 if not p.exists():errors.append('missing '+str(p));return
 if p.is_file():files[label]={'sha256':sha(p),'size':p.stat().st_size};return
 for folder,dirs,names in os.walk(p,followlinks=True):
  dirs[:]=[d for d in dirs if d!='__pycache__' and not (Path(folder)/d).resolve() in Path(folder).resolve().parents]
  for n in names:
   q=Path(folder)/n
   if q.suffix=='.pyc':continue
   rel=label+'/'+str(q.relative_to(p))
   try:
    if q.is_symlink():links[rel]=os.readlink(q)
    if q.is_file():files[rel]={'sha256':sha(q),'size':q.stat().st_size}
   except (OSError,ValueError) as e:errors.append([rel,str(e)])
for rel in ['compiler','tools/bisheng_compiler','lib64','include','conf']:
 tree(base/rel,'CANN/'+rel)
mods=['asc_op_compile_base','asc_opc_tool','op_compile_tool','opc_tool','te','tbe','te_fusion','superkernel','auto_tune','schedule_search']
site=base/'python/site-packages'
for m in mods:tree(site/m,'CANN/python/'+m)
for rel in ['reports/baseline-old1076-20261029/arm_nn_adapters','TileOPs/benchmarks/ascend_all_pairs/ops_nn','ascend_baselines/ops-nn/matmul/batch_mat_mul_v3','tools/baseline_equivalence_python.sh','tools/baseline_equivalence_entry.py']:
 tree(workspace/rel,'WORKSPACE/'+rel)
record_checks=[]
for dist in site.glob('*.dist-info'):
 if not any(dist.name.startswith(m+'-') for m in mods):continue
 rec=dist/'RECORD'
 if not rec.exists():continue
 checked=0;mismatch=[];missing=[]
 for row in csv.reader(rec.open()):
  if len(row)<2 or not row[1].startswith('sha256='):continue
  p=site/row[0]
  if not p.is_file():missing.append(row[0]);continue
  actual=base64.urlsafe_b64encode(bytes.fromhex(sha(p))).decode().rstrip('=');checked+=1
  if actual!=row[1][7:]:mismatch.append(row[0])
 record_checks.append({'dist':dist.name,'checked':checked,'mismatch':mismatch,'missing':missing})
versions={}
for rel in ['compiler/version.info','tools/bisheng_compiler/bin/bisheng']:
 p=base/rel
 if rel.endswith('.info'):versions[rel]=p.read_text()
 else:
  r=subprocess.run([str(p),'--version'],capture_output=True,text=True);versions[rel]=r.stdout+r.stderr
print(json.dumps({'at':time.time(),'files':files,'links':links,'errors':errors,'versions':versions,'installed_record_checks':record_checks,'unique_file_hashes':len(cache)}))
