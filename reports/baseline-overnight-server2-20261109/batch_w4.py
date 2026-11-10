import pathlib,subprocess,json
P=pathlib.Path(__file__).resolve().parent
rows=[]
for task in sorted((P/'tasks').glob('*-w4.json')):
 r=subprocess.run(['python3',str(P/'run_one.py'),task.name,'w4-v3','device'])
 rows.append({'task':task.name,'exit':r.returncode})
 (P/'W4_BATCH.json').write_text(json.dumps(rows,indent=2))
 if r.returncode:break
