import pathlib,subprocess,json
P=pathlib.Path(__file__).resolve().parent
rows=[]
for t in sorted((P/'tasks').glob('*-sampling-v4.json')):
 r=subprocess.run(['python3',str(P/'run_one_queue.py'),t.name,'sampling-v4','device'])
 rows.append({'task':t.name,'exit':r.returncode});(P/'SAMPLING_V4_BATCH.json').write_text(json.dumps(rows,indent=2))
 if r.returncode:raise SystemExit(r.returncode)
