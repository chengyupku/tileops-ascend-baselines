import pathlib,subprocess,tarfile,io,sys
p=pathlib.Path(__file__).resolve().parent;ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_linalg36'
if sys.argv[1]=='put':
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w') as t:
  for n in sys.argv[2:]:t.add(p/n,arcname=n)
 subprocess.run([ssh,'openlibing-DevEnv-055409','tar xf - -C '+remote],input=b.getvalue(),check=True)
else:
 r=subprocess.run([ssh,'openlibing-DevEnv-055409','tar cf - --ignore-failed-read -C '+remote+' runs '+ ' '.join(n for i in range(4,15) for n in ['batch%02d'%i,'batch%02d_progress.json'%i])],stdout=subprocess.PIPE,check=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(p,filter='data')
