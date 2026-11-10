import collections, hashlib, json, pathlib, platform, statistics, sys, time
sys.path.insert(0, 'tools')
from run_baseline_cache import quarantine_key
h=sys.argv[1]; R=pathlib.Path('reports/baseline-cache-20261027'); H=R/h; D=R/'inspection-20261029T001544'
read=lambda p:json.loads(p.read_text())
tasks=read(D/'server2-analysis.json')['ready']; ids={t['id'] for t in tasks}
rows=[json.loads(s) for s in (H/'journal.jsonl').read_text().splitlines() if s.strip()]
caches=[json.loads(s) for s in (H/'CACHE.jsonl').read_text().splitlines() if s.strip()]
q=read(H/'QUARANTINE.json'); metrics=collections.defaultdict(list)
for r in rows:
    if r.get('wall_seconds') and r['status'] not in ('resource_wait','temporarily_skipped_library'):
        metrics['::'.join(r['key'][:2])].append(r['wall_seconds'])
routes={r['id']:r for r in read(D/(h+'-CANDIDATE_ROUTES.json'))['other_not_classified']} if h!='server2' else {}
proof=[]
for t in tasks:
    hist=[r for r in rows if r['id']==t['id']]
    route=routes.get(t['id'],{})
    proof.append({'id':t['id'],'key':t['key'],'prior_statuses':[r['status'] for r in hist],
        'cached':any(r['task_id']==t['id'] for r in caches),'hold':quarantine_key(t,q),
        'route_observation':route.get('observation'),
        'route_source':[x for x in route.get('chain',[]) if 'ascend_all_pairs/' in x['file']]})
recent_native={}
for lib in ['sgl-kernel-npu','vllm-ascend']:
    successes=[r for r in caches if r['key'][0]==lib]
    if successes:
        last=successes[-1]; e=last['identity']['environment']; sources=last['identity']['sources']
        recent_native[lib]={'cached_count':len(successes),'latest_key':last['key'],
            'runtime':{k:e.get(k) for k in ['python','torch','torch_npu','cann_home','platform']},
            'native_files':{p:{'exists':pathlib.Path(p).is_file(),'expected':v} for p,v in sources.items() if p.endswith('.so')},
            'fresh_audit':str(D/(h+'-AUDIT.json'))}
result={'at':time.time(),'host':h,'architecture':platform.machine(),'tasks':proof,'recent_native':recent_native,
        'op_median_attempt_seconds':{k:statistics.median(v) for k,v in metrics.items() if k.startswith(('sgl-kernel-npu::','vllm-ascend::'))},
        'provider_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for lib in ['sgl','vllm'] for p in (pathlib.Path('TileOPs/benchmarks/ascend_all_pairs')/lib).glob('*.py')}}
(D/(h+'-MIGRATION_PREFLIGHT.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'host':h,'cached_collisions':sum(x['cached'] for x in proof),'prior_attempted':sum(any(s not in ('temporarily_skipped_library','resource_wait') for s in x['prior_statuses']) for x in proof),'held':sum(bool(x['hold']) for x in proof),'native':recent_native},ensure_ascii=False))
