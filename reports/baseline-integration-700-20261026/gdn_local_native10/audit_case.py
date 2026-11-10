import json,hashlib,re,sys,datetime
from pathlib import Path
W=Path(__file__).resolve().parent

def audit(case):
 d=W/case;r=json.loads((d/'checker.json').read_text());g=json.loads((d/'GUARD_RECEIPT.json').read_text());h=json.loads((d/'HBM_ADMISSION.json').read_text());contract=json.loads((W/'CONTRACTS.json').read_text())[case]
 assert g['returncode']==0 and r['status']=='verified' and r['inputs_preserved']
 assert r['verification']['checked_outputs']==r['verification']['total_outputs']==2
 assert not r['provider_source_changed'] and not r['oracle_source_changed']
 assert r['case_snapshot']['parameters']==contract['params']
 assert r['independent_cpu_reference'] and h['admitted']
 outputs=r['observed_calls']['candidate'][0]['outputs']['items']
 for actual,name in zip(outputs,['o','final_state']):
  expected=contract['params'][name];assert actual['shape']==expected[0] and actual['dtype']==expected[1]
 recipe=json.loads((W/'chunk_package/RUNTIME_RECIPE.json').read_text());loaded=json.loads((d/'loaded-libraries.json').read_text());identities={str(Path(x['path']).resolve()):x['sha256'] for x in loaded}
 libraries=[]
 for name in ['opapi','nnopbase']:
  p=Path(recipe[name]).resolve();sha=hashlib.sha256(p.read_bytes()).hexdigest();assert identities[str(p)]==sha;libraries.append({'role':name,'path':str(p),'sha256':sha})
 trace=d/'native-open.trace';selected=[]
 for i,line in enumerate(trace.open(),1):
  if 'ChunkGatedDeltaRule_' in line and '.o"' in line and 'O_RDONLY)' in line and not re.search(r'= -1',line):
   path=Path(line.split('"')[1]);assert str(path).startswith(recipe['vendor']+'/');selected.append({'trace_line':i,'line':line.strip(),'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 assert selected,'No actual loader .o open distinct from Python source-pin hash opens'
 receipt={'key':['ops-transformer','GDNFwdOp',case],'status':'verified_unregistered','accepted_added':0,'registered':False,'verification':r['verification'],'inputs_preserved':True,'outputs':outputs,'input_seed':r['input_seed'],'checker_seconds':r['wall_seconds'],'guard_elapsed_seconds':g['elapsed_seconds'],'libraries':libraries,'selected_device_objects':selected,'native_evidence':'Unmodified pinned DirectACL public GetWorkspaceSize/launch path returned successfully in canonical candidate; runtime loader O_RDONLY device object opens separate from Python O_CLOEXEC pin reads; actual library maps and hashes recorded. No standalone profiler launch count claimed.','hashes':{n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['checker.json','GUARD_RECEIPT.json','HBM_ADMISSION.json','fresh_resource_admission.json','loaded-maps.txt','loaded-libraries.json','native-open.trace','guard.stdout.log','guard.stderr.log','child.sh']}}
 (d/'AUDIT.json').write_text(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
 r=audit(sys.argv[1]);print(json.dumps({k:r[k] for k in ['key','status','verification','checker_seconds']}))
