import os,sys,json,hashlib,subprocess,time
from pathlib import Path
root=Path(sys.argv[1]);host=sys.argv[2];out={'host':host,'root':str(root),'at':time.time(),'adapter_trees':{},'source_repos':{}}
ignore={'.git','__pycache__','build','build_out','dist','.cache','.venv','node_modules'}
exts={'.py','.cpp','.cc','.c','.h','.hpp','.cuh','.cu','.asc','.cmake','.sh','.json','.yaml','.yml','.toml','.ini'}
def run(p,args):
 r=subprocess.run(['git','-C',str(p),*args],text=True,capture_output=True);return r.returncode,r.stdout

def hashes(p):
 files={}
 for d,dirs,names in os.walk(p,followlinks=False):
  dirs[:]=[x for x in dirs if x not in ignore]
  for n in names:
   q=Path(d)/n
   if q.suffix not in exts and n!='CMakeLists.txt':continue
   try:
    b=q.read_bytes();files[str(q.relative_to(p))]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'mtime':q.stat().st_mtime}
   except OSError:pass
 return files
base=root/'TileOPs/benchmarks/ascend_all_pairs'
if base.exists():
 for p in sorted(base.iterdir()):
  if p.is_dir() and p.name not in ignore:out['adapter_trees'][p.name]=hashes(p)
base=root/'ascend_baselines'
if base.exists():
 for p in sorted(base.iterdir()):
  if not p.is_dir():continue
  rc,top=run(p,['rev-parse','--show-toplevel']);git=rc==0 and Path(top.strip()).resolve()==p.resolve()
  rec={'resolved':str(p.resolve()),'git_present':git,'source_files':hashes(p)}
  if git:
   rec['head']=run(p,['rev-parse','HEAD'])[1].strip();rec['tracked_status']=run(p,['status','--short','--untracked-files=no'])[1];rec['diff']=run(p,['diff','HEAD','--'])[1];rec['untracked']=run(p,['ls-files','--others','--exclude-standard'])[1].splitlines();rec['last_commit']=run(p,['log','-1','--format=%H %aI %an %s'])[1].strip()
  out['source_repos'][p.name]=rec
print(json.dumps(out))
