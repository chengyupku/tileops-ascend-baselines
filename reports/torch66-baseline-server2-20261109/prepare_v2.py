import pathlib,subprocess,json,time
P=pathlib.Path(__file__).resolve().parent
for i in range(42,66):
 out=P/'runs'/f'{i:02d}'/'v2';out.mkdir(parents=True,exist_ok=True)
 if (out/'PREPARED.json').exists():continue
 with (out/'prepare.log').open('w') as log:r=subprocess.run(['bash',str(P/'launch_v2.sh'),'prepare',str(P/'tasks_v2'/f'{i:02d}.json'),str(out)],stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps({'ordinal':i,'prepare_exit':r.returncode,'at':time.time()}),flush=True)
