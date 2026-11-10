"""Offline SHA/raw replay; pilot results are explicitly not formal acceptance."""
import pathlib,json,importlib.util,statistics,hashlib,csv
P=pathlib.Path(__file__).resolve().parent;R=P/'remote';REMOTE='/opt/quant-baseline-fix-server2-20261110'
parserpath=P.parent/'torch-all2208-20261105/tools/kernel_task_profile_full32.py'
spec=importlib.util.spec_from_file_location('parser',parserpath);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tasks=json.loads((P/'TASKS.json').read_text());rows=[]
for f in sorted((R/'runs').glob('*/*/*/result.json')):
    m=json.loads(f.read_text());idx=int(f.parents[2].name);variant=f.parents[1].name;attempt=f.parent.name
    row={'index':idx,'op':tasks[idx]['op'],'case':tasks[idx]['case'],'variant':variant,'attempt':attempt,'status':m['status'],'result_sha256':sha(f)}
    try:
        if 'timing' in m:
            mode='pilot' if m['windows']==1 else 'formal';gp=R/f'{mode}-{idx}-{variant}-{attempt}-guard.json';
            if not gp.exists():
                matches=[]
                for batch in R.glob('batch*.json'):
                    if '.progress.' in batch.name or '-guard' in batch.name:continue
                    items=json.loads(batch.read_text())
                    if isinstance(items,list) and any(j['index']==idx and j['variant']==variant and j['attempt']==attempt for j in items):matches.append(batch)
                assert len(matches)==1,('batch guard mapping',matches)
                b=matches[0];gp=R/(b.stem+'-guard.json');progress=json.loads((R/(b.stem+'.progress.json')).read_text())
                assert any(j.get('exit_code')==0 and j['job']['index']==idx and j['job']['variant']==variant and j['job']['attempt']==attempt for j in progress)
            g=json.loads(gp.read_text())
            assert g['performance_valid'] and g['workload_exit_code']==0 and g['phase']=='finished'
            assert m['hardware']['cube']==32 and m['hardware']['vector']==64
            assert sha(f.parent/'correctness.json')==m['correctness_sha256']
            assert sha(f.parent/'SOURCE.json')==m['source_sha256']
            t=m['timing'];samples=[];count=[];aicpu=[]
            for prof in t['retained_profiles']:
                path=R/pathlib.Path(prof['profile_path']).relative_to(REMOTE)
                assert parser.trace_digest(path)==prof['profile_trace_sha256']
                x=parser.parse_profile(path,1)
                assert x['profile_kernel_tasks']==prof['profile_kernel_tasks']
                assert x['profile_named_task_coverage_complete']
                samples.append(x['profile_kernel_us']/1000);count.append(x['profile_kernel_count']);aicpu.append(x['profile_aicpu_task_count'])
            assert len(samples)==m['windows']
            assert abs(statistics.mean(samples)-t['latency_ms'])<1e-9
            assert all(abs(a-b)<1e-9 for a,b in zip(samples,t['latency_samples_ms']))
            row.update(status='formal_raw_verified' if m['windows']==15 else 'pilot_raw_verified',us=1000*t['latency_ms'],windows=len(samples),task_counts=count,aicpu_counts=aicpu,old_web_torch_us=tasks[idx]['latency_us'],guard_sha256=sha(gp))
        elif m['status']=='edge_checked_pending_guard':
            e=json.loads((P/'EDGE_AUDIT.json').read_text())
            item=next(x for x in e['checks'] if x['job']['index']==idx and x['job']['variant']==variant and x['job']['attempt']==attempt)
            assert item['result_sha256']==sha(f)
            assert all(x['passed'] for x in m['checks'])
            row.update(status='edge_verified',edge_check_count=len(m['checks']))
        else:
            row['error']=m.get('error','')[-1200:];row['phase']=m.get('phase')
    except Exception as e:row.update(status='audit_failed',audit_error=repr(e))
    rows.append(row)
(P/'RESULTS.json').write_text(json.dumps(rows,indent=2))
cols=['index','op','case','variant','attempt','status','us','windows','old_web_torch_us','task_counts','aicpu_counts','error','audit_error']
with (P/'RESULTS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
from collections import Counter
print(json.dumps(dict(Counter(r['status'] for r in rows))))
