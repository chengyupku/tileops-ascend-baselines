import json,time,subprocess,base64
from pathlib import Path
h=Path('reports/baseline-retest32-20261030/server1');p=Path('tools/run_baseline_retest32_server1.py');s=p.read_text();(h/'runner.before_v45.py').write_text(s)
s=s.replace('key=lambda t: (',"key=lambda t: (t['id']!='7edf2980112b388ff645',",1).replace('        return (',"        return (x['id']=='7edf2980112b388ff645' and x.get('revision')=='prepermute_original_graph_v44' and x.get('error') is None) or (",1).replace("'timeout':max(t['timeout'],2202)","'timeout':1200 if t['id']=='7edf2980112b388ff645' else max(t['timeout'],2202)").replace("if len(chosen)==4 or","if len(chosen)==4 or t['id']=='7edf2980112b388ff645' or",1).replace("'revision':'prepermute_original_graph_v44'","'revision':'gdn_profile_timeout_v45'");compile(s,str(p),'exec');p.write_text(s)
row={'at':time.time(),'revision':'gdn_profile_timeout_v45','id':'7edf2980112b388ff645','new_condition':'v44 full correctness passed, reached profile phase at deadline; guard exit124/workload143, no conflict. Raise exactkey device timeout706->1200; original checker/profile/kernel unchanged'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='data='+repr(base64.b64encode(s.encode()).decode())+'\nrow='+repr(row)+'''
import json,time,base64
from pathlib import Path
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1';p=r/'tools/run_baseline_retest32_server1.py';(h/'runner.before_v45.py').write_bytes(p.read_bytes());q=p.with_suffix('.deploy');q.write_bytes(base64.b64decode(data));q.replace(p)
(h/'STOP').write_text(json.dumps({'reason':'drain for GDN timeout v45','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
'''
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
