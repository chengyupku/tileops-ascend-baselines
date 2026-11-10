import pathlib,subprocess,json
p=pathlib.Path(__file__).resolve().parent
rows=[]
for i in range(139,144):
 label='sampling-v4-r2' if i==139 else 'sampling-v4'
 r=subprocess.run(['python3',str(p/'run_one_queue.py'),f'{i}-sampling-v4.json',label,'device'])
 rows.append({'ordinal':i,'label':label,'exit':r.returncode});(p/'SAMPLING_V4_RESUME_1718.json').write_text(json.dumps(rows,indent=2))
 if r.returncode:raise SystemExit(r.returncode)
