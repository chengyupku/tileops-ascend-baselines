import pathlib,json,subprocess,concurrent.futures,datetime,shlex
P=pathlib.Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code='''import pathlib,json,datetime
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
  if not any(pathlib.Path(x).name in ['npu-guard','npu_guard.py'] for x in a):continue
  if 'run' not in a or not any('baseline700-20261026/surge_tla_attention97' in x for x in a):continue
  st=(p/'stat').read_text().rsplit(')',1)[1].split()
  rows.append({'pid':int(p.name),'ppid':int(st[1]),'pgid':int(st[2]),'state':st[0],'start_ticks':st[19],'argv':a})
 except (OSError,ValueError):pass
print(json.dumps({'host_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'guards':rows}))'''
def probe(host):
 cmd=['python3','-c',code] if host=='local' else [SSH,host,'python3 -c '+shlex.quote(code)]
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
  return {'host':host,'exit_code':r.returncode,'sample':json.loads(r.stdout) if r.returncode==0 else None,'error':r.stderr[-600:]}
 except Exception as e:return {'host':host,'error':str(e),'sample':None}
with concurrent.futures.ThreadPoolExecutor() as ex:rows=list(ex.map(probe,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'CPU-only /proc argv and PID start-identity census; no device imports or device access','hosts':rows}
(P/'ACTIVE_GUARD_CENSUS.json').write_text(json.dumps(x,indent=2));print(json.dumps([{'host':r['host'],'error':r.get('error'),'guards':(r.get('sample') or {}).get('guards')} for r in rows]))
