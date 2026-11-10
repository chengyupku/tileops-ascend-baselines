"""CPU-only minimal private deployment; existing compiler/SDK/runtime are reused."""
import json,pathlib,subprocess,shlex,tarfile,io,sys,datetime
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2]
ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_pto_registration23'
names=['launch.py','registration.py','runtime_entry.py','preflight.py','build_contracts.py','cpu_checks.py','CONTRACTS.json','HOSTS.json','LOCAL_PINS.json','DATA_HASHES.json','EVIDENCE.json','setup_local.py','setup_server1.py','setup_server2.py']
names += [str(f.relative_to(P)) for f in (P/'closure').rglob('*') if f.is_file()]
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for n in names:t.add(P/n,arcname=n)
rows=[]
for host,alias in [('server1','openlibing-DevEnv-661495'),('server2','openlibing-DevEnv-055409')]:
 r=subprocess.run([str(ssh),alias,'mkdir -p '+shlex.quote(remote)+' && tar -xzf - -C '+shlex.quote(remote)],input=buf.getvalue(),capture_output=True,timeout=45)
 assert r.returncode==0,r.stderr
 python='/opt/tileops-ci/python/cpython-3.12.15-linux-aarch64-gnu/bin/python3'
 argv=[python,'-I','-S','-B',remote+'/preflight.py']
 r=subprocess.run([str(ssh),alias,shlex.join(argv)],capture_output=True,text=True,timeout=55)
 (P/(host+'_preflight.stdout')).write_text(r.stdout);(P/(host+'_preflight.stderr')).write_text(r.stderr)
 rows.append(dict(host=host,argv=argv,exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
 if r.stdout.strip():
  try:(P/('PREFLIGHT_'+host+'.json')).write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')
  except ValueError:pass
for row in rows:
 host=row['host'];alias={'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[host]
 argv=[python,'-I','-S','-B',remote+'/cpu_checks.py']
 r=subprocess.run([str(ssh),alias,shlex.join(argv)],capture_output=True,text=True,timeout=55)
 row['cpu_route']={'argv':argv,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 (P/('CPU_ROUTE_'+host+'.json')).write_text(json.dumps(row['cpu_route'],indent=2)+'\n')
(P/'DEPLOYMENT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tar_bytes=len(buf.getvalue()),files=len(names),rows=rows),indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['host','exit','stderr']} for r in rows]))
