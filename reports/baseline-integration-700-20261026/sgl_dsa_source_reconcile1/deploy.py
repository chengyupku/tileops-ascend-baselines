import pathlib,json,subprocess,ast
P=pathlib.Path(__file__).resolve().parent
module=(P/'integration700_sgl_dsa_source_reconcile1.py').read_text();tree=ast.parse(module)
pins=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='PINS')
files={n:(P/n).read_text() for n in ['dsa_adapter.py','checker.original.json','SOURCE_RECONCILIATION.json','provider_init.before.py','provider_init.after_race.py','SOURCE_RACE.diff']}
payload={'workspace':'/opt/tileops-all-pairs/workspace','remote':True,'files':files,'module':module,'append':(P/'provider_registration.append.py').read_text(),'pins':pins,'params':json.loads((P/'checker.original.json').read_text())['case_snapshot']['parameters']}
script=(P/'deploy_host.py').read_text();command="python3 -c "+__import__('shlex').quote(script)
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r=subprocess.run([SSH,'openlibing-DevEnv-055409',command],input=json.dumps(payload),text=True,capture_output=True,timeout=60)
(P/'REMOTE_DEPLOY.stdout').write_text(r.stdout);(P/'REMOTE_DEPLOY.stderr').write_text(r.stderr);print('REMOTE rc',r.returncode);print(r.stdout[:700]);print(r.stderr)
if r.returncode:raise SystemExit(r.returncode)
(P/'FORMAL_REGISTRY_PROOF.server2.json').write_text(r.stdout)
payload.update(workspace=str(P.parents[2]),remote=False)
r=subprocess.run(['python3',str(P/'deploy_host.py')],input=json.dumps(payload),text=True,capture_output=True,timeout=60)
(P/'LOCAL_DEPLOY.stdout').write_text(r.stdout);(P/'LOCAL_DEPLOY.stderr').write_text(r.stderr);print('LOCAL rc',r.returncode);print(r.stdout[:700]);print(r.stderr)
raise SystemExit(r.returncode)
