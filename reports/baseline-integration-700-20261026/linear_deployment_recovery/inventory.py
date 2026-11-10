import os,json,pathlib,time,hashlib
root=pathlib.Path('/opt/tileops-all-pairs/workspace'); out=root/'reports/baseline-integration-700-20261026/linear_deployment_recovery';out.mkdir(parents=True,exist_ok=True)
found=[]; errors=[]; start=time.time()
for base in ['/usr/local/Ascend',str(root/'.cache'),'/opt/tileops-ci']:
 for d,ds,fs in os.walk(base,onerror=lambda e:errors.append(str(e))):
  ds[:]=[x for x in ds if x not in ['.git','node_modules','__pycache__']]
  for f in fs:
   low=f.lower()
   if f in ['ascend_all_cann_install.info','ascend_toolkit_install.info'] or (('causal' in low or 'chunk_kda' in low or 'chunkkda' in low) and low.endswith(('.json','.so','.run','.o','.sh'))) or ('cann' in low and low.endswith('.run')):
    p=pathlib.Path(d)/f
    found.append(dict(path=str(p),size=p.stat().st_size,realpath=str(p.resolve())))
r=dict(elapsed=time.time()-start,files=found,errors=errors);(out/'CPU_INVENTORY.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
