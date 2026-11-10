"""CPU audit of measured cases, source identities and guarded requests."""
import csv
import hashlib
import json
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OPS={'LogFwdOp':'log','Log1pFwdOp':'log1p','SqrtFwdOp':'sqrt','RsqrtFwdOp':'rsqrt'}
PLAN=json.loads((HERE/'PLAN.json').read_text())
FORMAL_LIBRARIES=set(PLAN['baseline_scope'])
helper=ROOT/'TileOPs/src/tileops/kernels/ascend/elementwise/_logroots.py'
helper_hash=hashlib.sha256(helper.read_bytes()).hexdigest()
guards={}
requests=[]
for p in (HERE/'requests').glob('*/status.json'):
    row=json.loads(p.read_text());requests.append(row)
    command=row['command']
    if '--output' in command:
        output=command[command.index('--output')+1]
        guards[output]=row

def relative(path):return str(path.relative_to(ROOT))
def qualified(row):
    samples=row.get('latency_samples_ms',[])
    return (row.get('correct') is True and row.get('input_unchanged') is True
            and row.get('measurement_protocol')=='ascend950-isolated-ffts-v2'
            and len(samples)==15 and all(math.isfinite(x) and x>0 for x in samples)
            and math.isclose(sum(samples)/15,row['latency_ms'],rel_tol=1e-9))

own={};rejected=[];baselines={};failures=[];attempted=set()
for p in (HERE/'final').glob('final-*.jsonl'):
    guard=guards.get(relative(p))
    for line in p.read_text().splitlines():
        row=json.loads(line);key=(row['op'],row['case_id'])
        module=helper.with_name(OPS[row['op']]+'.py')
        valid=(guard and guard['guard_exit_code']==0 and qualified(row)
               and row.get('helper_hash')==helper_hash
               and row.get('source_hash')==hashlib.sha256(module.read_bytes()).hexdigest())
        if valid:
            if key not in own or row['time'] >= own[key]['time']:
                own[key]={**row,'evidence':str(p.relative_to(HERE)),'request':guard['label']}
        else:rejected.append({'file':str(p),'key':key,'reason':'guard/source/protocol gate'})
for p in (HERE/'baseline').glob('baseline-*.jsonl'):
    guard=guards.get(relative(p))
    for line in p.read_text().splitlines():
        row=json.loads(line);key=(row['op'],row['case_id'])
        attempted.add((*key,row['library']))
        if row.get('status')!='measured' or not qualified(row) or not guard or guard['guard_exit_code']!=0:
            failures.append({'op':row['op'],'case_id':row['case_id'],'library':row['library'],'status':row.get('status'),'error':row.get('error'),'guard':guard and guard['guard_exit_code'],'file':str(p.relative_to(HERE))})
            continue
        receipt=Path(row['environment_receipt'])
        receipt=HERE/'environment'/receipt.name
        if not receipt.exists():
            rejected.append({'file':str(p),'key':key,'reason':'missing environment receipt'});continue
        env=json.loads(receipt.read_text())
        if env.get('host')!='a5d0b50dc68f':
            rejected.append({'file':str(p),'key':key,'reason':'different host'});continue
        if key in own and row['input_digest']!=own[key]['input_digest']:
            rejected.append({'file':str(p),'key':key,'reason':'different actual inputs'});continue
        prior=baselines.setdefault(key,{}).get(row['library'])
        if prior is None or guard['completed_at'] >= prior['request_completed_at']:
            baselines[key][row['library']]={**row,'evidence':str(p.relative_to(HERE)),'request':guard['label'],'request_completed_at':guard['completed_at']}

comparison=[]
expected=set()
for case in json.loads((HERE/'catalog.json').read_text()):
    key=(case['operator'],case['case_id']);actual=own.get(key);all_candidates=baselines.get(key,{})
    candidates={k:v for k,v in all_candidates.items() if k in FORMAL_LIBRARIES}
    libraries=[lib for lib in ('ops-math','ops-nn','torch','torch-compile') if lib in FORMAL_LIBRARIES]
    if key[0]!='Log1pFwdOp':libraries.append('tilelang-ascend')
    if key[0] in {'LogFwdOp','SqrtFwdOp'}:libraries.append('catlass')
    expected.update((*key,library) for library in libraries)
    winner=min(candidates.values(),key=lambda x:x['latency_ms']) if candidates else None
    all_winner=min(all_candidates.values(),key=lambda x:x['latency_ms']) if all_candidates else None
    ratio=winner['latency_ms']/actual['latency_ms'] if winner and actual else None
    all_ratio=all_winner['latency_ms']/actual['latency_ms'] if all_winner and actual else None
    comparison.append({'op':key[0],'case_id':key[1],'tileops_us':actual['latency_ms']*1000 if actual else None,
                       'best_library':winner['library'] if winner else None,
                       'best_us':winner['latency_ms']*1000 if winner else None,
                       'ratio':ratio,'pass':ratio is not None and ratio>=.8,'valid_baselines':len(candidates),
                       'best_including_additional_library':all_winner['library'] if all_winner else None,
                       'best_including_additional_us':all_winner['latency_ms']*1000 if all_winner else None,
                       'ratio_including_additional':all_ratio,'pass_including_additional':all_ratio is not None and all_ratio>=.8})
with (HERE/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
summary={'own_valid_cases':len(own),'valid_baseline_pairs':sum(map(len,baselines.values())),
         'formal_valid_baseline_pairs':sum(lib in FORMAL_LIBRARIES for libs in baselines.values() for lib in libs),
         'additional_valid_baseline_pairs':sum(lib not in FORMAL_LIBRARIES for libs in baselines.values() for lib in libs),
         'expected_baseline_pairs':len(expected),
         'baseline_pairs_unattempted':sorted(expected-attempted),
         'baseline_pairs_without_valid_measurement':sorted(expected-{(*key,lib) for key,libs in baselines.items() for lib in libs}),
         'cases_passing':sum(r['pass'] for r in comparison),'total_cases':len(comparison),
         'cases_passing_including_additional':sum(r['pass_including_additional'] for r in comparison),
         'minimum_ratio_including_additional':min((r['ratio_including_additional'] for r in comparison if r['ratio_including_additional'] is not None),default=None),
         'minimum_ratio':min((r['ratio'] for r in comparison if r['ratio'] is not None),default=None),
         'by_op':{op:{'pass':sum(r['pass'] for r in comparison if r['op']==op),'min_ratio':min((r['ratio'] for r in comparison if r['op']==op and r['ratio'] is not None),default=None)} for op in OPS},
         'baseline_failures':failures,'rejected':rejected,'helper_sha256':helper_hash,
         'timing':{'completed_requests':len(requests),'queue_and_idle_check_seconds':sum(r.get('queue_and_idle_check_seconds') or 0 for r in requests),'reservation_seconds':sum(r.get('reservation_seconds') or 0 for r in requests)},
         'comparison':comparison}
(HERE/'audit.json').write_text(json.dumps(summary,indent=2))
(HERE/'qualified-baselines.json').write_text(json.dumps([r for rs in baselines.values() for r in rs.values()],indent=2))
print(json.dumps({**{k:v for k,v in summary.items() if k not in ('comparison','baseline_failures','rejected','baseline_pairs_unattempted','baseline_pairs_without_valid_measurement')},
                  'unattempted_pair_count':len(expected-attempted),
                  'unqualified_pair_count':len(summary['baseline_pairs_without_valid_measurement'])},indent=2))
