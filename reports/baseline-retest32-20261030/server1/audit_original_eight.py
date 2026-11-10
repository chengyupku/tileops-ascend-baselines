"""Read-only remote acceptance audit for the eight pending pairs raised by user."""
import json,subprocess
from pathlib import Path
IDS=['a8f02491d524878bd7e7','1b13f732539cead3d76c','d39eeba69330b75b72ab','d02e991de94765207b84','246f93d26b8391bddaf2','235ddc5395c16406b07b','f972002c7db9f4c03ebc','54ea178d41092467b54e']
code='ids='+repr(IDS)+r'''
import json,time,hashlib
from pathlib import Path
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1')
latest={x['id']:x for x in map(json.loads,(h/'journal.jsonl').read_text().splitlines())}
result=[]
for ident in ids:
 j=latest[ident];m=json.loads(Path(j['measurement']).read_text());g=json.loads(Path(j['guard']).read_text())
 cpath=Path(m.get('correctness_receipt',Path(j['measurement']).parent/'correctness.json'))
 c=json.loads(cpath.read_text()) if cpath.exists() else {}
 ok=(j['status']=='cached' and m['status']=='cached' and m.get('performance_valid') is True and g.get('performance_valid') is True and g.get('workload_exit_code')==0 and c.get('status')=='verified' and c.get('inputs_preserved') is True and not c.get('provider_source_changed') and not c.get('oracle_source_changed') and m.get('identity',{}).get('environment',{}).get('aicore_count')==32 and hashlib.sha256(cpath.read_bytes()).hexdigest()==m.get('correctness_sha256') and bool(m.get('guard_binding')))
 result.append({'id':ident,'key':j['key'],'accepted':ok,'measurement':j['measurement'],'guard':j['guard'],'correctness':str(cpath),'revision':j.get('revision')})
print(json.dumps({'at':time.time(),'all_eight_accepted':all(x['accepted'] for x in result),'accepted_count':sum(x['accepted'] for x in result),'pairs':result}))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True)
audit=json.loads(r.stdout)
Path(__file__).with_name('ORIGINAL_EIGHT_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'all_eight_accepted':audit['all_eight_accepted'],'accepted_count':audit['accepted_count']}))
