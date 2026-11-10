import json,pathlib,subprocess,shlex,hashlib,runpy
P=pathlib.Path(__file__).resolve().parent
rows=json.load(open(P/'ADD_GEMM_RECEIPTS.json'));wanted={}
for r in rows.values():
 if r['key'][0] not in ('catlass','ops-tensor'):continue
 for path,sha in r['sources'].items():
  if path.endswith('.so') and ('/catlass/native/' in path or '/ops_tensor/' in path):wanted[(path,sha)]=r['key'][0]
fn=runpy.run_path(str(P/'elf_functions.py'))['functions'];out=[]
for (path,sha),lib in wanted.items():
 dest=P/'verified_binaries'/(sha+'.so');remote='/opt/tileops-baseline-cache/workspace/'+path
 if not dest.exists():
  with dest.open('wb') as f:p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','cat '+shlex.quote(remote)],stdout=f,stderr=subprocess.PIPE)
 actual=hashlib.sha256(dest.read_bytes()).hexdigest();fs=fn(dest) if actual==sha else []
 row={'library':lib,'historical_path':path,'sha256':sha,'actual_sha256':actual,'matched':actual==sha,'local_path':str(dest),'functions':fs};out.append(row)
 print(lib,path,actual==sha,len(fs),flush=True)
(P/'MATCHED_NATIVE_BINARIES.json').write_text(json.dumps(out,indent=2))
