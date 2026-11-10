#!/opt/tileops-ci/runtime/bin/python3
"""Resume only proven completed original recipes through GNU make -o.
Never change an original recipe, source, tiling key or compiler option.
"""
import os,sys,pathlib,json,hashlib,shlex,time,shutil
R=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
args=[];skip=False
for a in sys.argv[1:]:
 if skip:skip=False;continue
 if a in ('-j','--jobs'):skip=True;continue
 if a.startswith('-j') or a.startswith('--jobs='):continue
 args.append(a)
reuse=[]; reason='not generated compiler make'
if '-f' in args:
 m=pathlib.Path(args[args.index('-f')+1])
 if m.is_file() and '_tmp_' in m.name and m.suffix=='.mk':
  reason='no matching checkpoint'
  archive=R/'generated_makefiles';archive.mkdir(exist_ok=True)
  saved=archive/(sha(m)+'.mk')
  if not saved.exists():shutil.copy2(m,saved)
  lines=m.read_text().splitlines(); rules={}
  for i,l in enumerate(lines):
   if l.startswith('\t'):
    try:
     a=shlex.split(l);out=a[a.index('-o')+1]
    except (ValueError,IndexError):continue
    if not out.endswith('.o') or not lines[i-1].endswith(':'):continue
    rules[sha_text:=hashlib.sha256(l.encode()).hexdigest()]=(lines[i-1][:-1],out)
  for cp in sorted((R/'checkpoints').glob('*/manifest.json')):
   c=json.loads(cp.read_text())
   if not all(pathlib.Path(p).is_file() and sha(p)==h for p,h in c['inputs_sha256'].items()):continue
   reason='exact original recipes, preprocessed inputs, generated source/header and compiler match'
   for entry in c['completed']:
    key=entry['recipe_sha256']
    if key not in rules:continue
    target,out=rules[key]
    if target in reuse:continue
    cached=cp.parent/entry['saved']
    if sha(cached)!=entry['sha256']:raise RuntimeError('checkpoint object changed')
    dest=pathlib.Path(out)
    if not dest.exists() or sha(dest)!=entry['sha256']:shutil.copy2(cached,dest)
    args+=['-o',target];reuse.append(target)
  log=R/'checkpoint_make_receipts';log.mkdir(exist_ok=True)
  (log/f'{time.time_ns()}.json').write_text(json.dumps({'makefile':str(m),'makefile_sha256':sha(m),'reuse_count':len(reuse),'targets':reuse,'reason':reason,'jobs':8},indent=2))
os.execv('/usr/bin/make',['/usr/bin/make',*args,'-j8'])
