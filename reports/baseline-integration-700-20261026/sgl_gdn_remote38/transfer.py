import pathlib,subprocess,tarfile,io,sys
P=pathlib.Path(__file__).resolve().parent;remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_gdn_remote38';ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';host='openlibing-DevEnv-055409'
if sys.argv[1]=='push':
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w') as t:
  for p in P.iterdir():
   if p.is_file() and p.name not in ("FEEDBACK.txt","transfer.py"):t.add(p,arcname=p.name)
 subprocess.run([ssh,host,f'mkdir -p {remote} && tar -xf - -C {remote}'],input=b.getvalue(),check=True)
else:
 r=subprocess.run([ssh,host,f'tar --exclude="./FEEDBACK.txt" --exclude="./transfer.py" --exclude="*/ASCEND_PROCESS_LOG_PATH" --exclude="*/ASCEND_CACHE_PATH" --exclude="*/tmp" --exclude="*/XDG_CACHE_HOME" --exclude="*/TRITON_CACHE_DIR" --exclude="*/TORCH_EXTENSIONS_DIR" --exclude="*/ASCEND_WORK_PATH" -cf - -C {remote} .'],stdout=subprocess.PIPE,check=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(P,filter='data')
