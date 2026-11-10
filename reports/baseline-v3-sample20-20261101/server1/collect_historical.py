import pathlib,json,subprocess
P=pathlib.Path(__file__).resolve().parent;rs=json.loads((P/'SAMPLE.json').read_text())['rows'];out={}
for host,alias in [('server1','openlibing-DevEnv-661495'),('server2','openlibing-DevEnv-055409')]:
 selected=[r for r in rs if r['host']==host]
 code='import pathlib,json\nrows='+repr(selected)+'\nout={}\nfor r in rows:\n p=pathlib.Path(r["correctness_receipt"]);out[r["task_id"]]=json.loads(p.read_text())\nprint(json.dumps(out))\n'
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',alias,'python3 -'],input=code,text=True,capture_output=True,check=True);out.update(json.loads(r.stdout))
(P/'HISTORICAL_CORRECTNESS.json').write_text(json.dumps(out,indent=2))
print('collected',len(out))
