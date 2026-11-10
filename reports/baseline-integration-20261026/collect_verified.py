import pathlib,json,subprocess,sys,os,shutil
root=pathlib.Path(sys.argv[1]); scope=root/'.cache/all-pairs/full_rollout_20261024'
files=[]
for base,dirs,names in os.walk(scope):
 dirs[:]=[x for x in dirs if x not in ['.git','__pycache__','cache','torch-cache','build','artifacts','tmp','site-packages']]
 files.extend(str(pathlib.Path(base)/x) for x in names if x=='RESULT.json' or ('checker' in x and x.endswith('.json')))
r=type('Result',(),{'stdout':'\n'.join(files)})()
out=[]
for line in r.stdout.splitlines():
 p=pathlib.Path(line)
 try:d=json.loads(p.read_text())
 except Exception:continue
 if not isinstance(d,dict):continue
 if d.get('library') and d.get('op') and d.get('case_id') and d.get('status')=='verified':out.append({'key':[d['library'],d['op'],d['case_id']],'path':line,'kind':'checker_verified'})
 for row in d.get('rows',[]) if isinstance(d.get('rows'),list) else []:
  if isinstance(row,dict) and row.get('status')=='verified' and row.get('key'):out.append({'key':row['key'],'path':line,'kind':'result_verified','checker':row.get('checker_receipt'),'guard_exit_code':row.get('guard_exit_code')})
print(json.dumps({'receipts':out,'files_scanned':len(r.stdout.splitlines())},ensure_ascii=False))
