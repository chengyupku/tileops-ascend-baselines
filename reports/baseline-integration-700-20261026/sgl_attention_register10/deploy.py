import pathlib,json,subprocess
P=pathlib.Path(__file__).resolve().parent;R=P.with_name('sgl_attention_recovery32');s=P/'stage'
payload={'remote_hashes':json.loads((P/'REMOTE_HASH_REQUEST.json').read_text()),'files':{'dsa_adapter.py':(s/'dsa_adapter.py').read_text()},'module':(s/'integration700_sgl_attention_register10.py').read_text(),'append':(s/'provider_registration.append.py').read_text(),'source_hashes':json.loads((R/'delta07/SOURCE_HASHES.json').read_text()),'keys':[r['case_id'] for r in json.loads((P/'SCOPE.json').read_text())]}
# Script and payload via stdin, no remote temp writes outside private report.
script=(P/'deploy_remote.py').read_text();import base64
cmd='python3 -B -c '+__import__('shlex').quote('import base64;exec(base64.b64decode('+repr(base64.b64encode(script.encode()).decode())+'))')
r=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409',cmd],input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(P/'deploy.stdout').write_bytes(r.stdout);(P/'deploy.stderr').write_bytes(r.stderr)
print('exit',r.returncode,r.stderr.decode()[-2500:])
if r.returncode==0:(P/'DEVICE_FREE_CHECKS.json').write_bytes(r.stdout);print('registration verified')
raise SystemExit(r.returncode)
