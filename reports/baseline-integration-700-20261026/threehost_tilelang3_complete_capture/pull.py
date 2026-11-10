import pathlib,subprocess,tarfile,json
P=pathlib.Path(__file__).resolve().parent;S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/'+P.name
cmd="cd "+R+" && tar czf - --exclude='runs/*/cache' runs FREEZE.json pre_*.json post_*.json"
with (P/'server1_evidence.tar.gz').open('wb') as f:subprocess.run([S,'openlibing-DevEnv-661495',cmd],stdout=f,check=True)
with tarfile.open(P/'server1_evidence.tar.gz') as t:t.extractall(P/'server1_evidence',filter='data')
print('evidence pulled')
