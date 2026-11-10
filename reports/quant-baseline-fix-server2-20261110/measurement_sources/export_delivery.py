"""Update the local baseline repository only. No git push, raw copy or site changes."""
import pathlib,json,hashlib,shutil
P=pathlib.Path(__file__).resolve().parent
repo=P.parents[1]/'exports/tileops-ascend-baselines'
campaign=P.name;target=repo/'reports'/campaign
coverage=json.loads((P/'COVERAGE.json').read_text())
assert not [r for r in coverage if r['status']=='pending'],'Formal campaign still has pending admitted work'
comparison=json.loads((P/'COMPARISON_SUMMARY.json').read_text());assert comparison['formal_comparable']==71
verified=json.loads((P/'delivery/verified_cases.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
source_files=list((P/'delivery').rglob('*.py'))+list((P/'delivery').rglob('*.json'))
exported=[]
for f in source_files:
 rel=f.relative_to(P/'delivery');out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(f.read_bytes());exported.append((f,out))
for name in ['REPORT.json','REPORT.txt','COVERAGE.json','ADAPTATION_CONTRACT.json','COMPILED_NATIVE_CONTRACT.json','FORMAL_GATE.json','REGISTRY_CPU_CHECKS.json']:
 f=P/name;out=target/name;out.write_bytes(f.read_bytes());exported.append((f,out))
# Frozen launch/provider/measurement source history, never live state or payloads.
for f in sorted(P.glob('*.py'))+sorted(P.glob('*.sh')):
 out=target/'measurement_sources'/f.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(f.read_bytes());exported.append((f,out))
rows=json.loads((repo/'catalog/results.json').read_text())
assert not any(r['campaign']==campaign for r in rows),'Do not overwrite an existing published campaign snapshot'
for v in verified:
 key=[v['variant'],v['op'],v['case']];identity=hashlib.sha256(json.dumps([campaign,key,v['attempt'],v['result_sha256']]).encode()).hexdigest()[:20]
 rows.append({'campaign':campaign,'id':identity,'key':key,'status':'accepted','attempt':v['attempt'],'case':{'op':v['op'],'case_id':v['case'],'parameters':v['parameters']},'raw_included':False,'result_sha256':v['result_sha256'],'guard_sha256':v['guard_sha256'],'source_manifest_sha256':v['source_manifest_sha256'],'measurement_summary':{'latency_ms':v['latency_us']/1000,'raw_windows':15,'kernel_task_counts':v['kernel_task_counts'],'aicpu_task_counts':v['aicpu_task_counts'],'protocol':'ascend950-isolated-ffts-v3'}})
labels={'original':'torch-eager-control','fixed':'torch-eager-layout-fixed','native':'torch-npu-antiquant','compile':'torch-compile-inductor-ascendc','compile_native':'torch-compile-inductor-antiquant','ops_math':'ops-math-native-broadcast','tilelang':'tilelang-no-identity-pad-candidate'}
for entry in coverage:
 if entry['status']=='formal_verified':continue
 key=[labels.get(entry['variant'],entry['variant']),entry['op'],entry['case']]
 identity=hashlib.sha256(json.dumps([campaign,key,entry['status']]).encode()).hexdigest()[:20]
 params=json.loads((P/f"remote/runs/{entry['index']}/prepare.json").read_text())['params']
 rows.append({'campaign':campaign,'id':identity,'key':key,'status':entry['status'],'attempt':None,'case':{'op':entry['op'],'case_id':entry['case'],'parameters':params},'raw_included':False,'reason':entry['reason'],'attempts':entry['attempts']})
put(repo/'catalog/results.json',rows)
campaigns=json.loads((repo/'catalog/campaigns.json').read_text());assert campaign not in campaigns;campaigns.append(campaign);put(repo/'catalog/campaigns.json',sorted(campaigns))
manifest=json.loads((repo/'catalog/source-manifest.json').read_text());known={r['path'] for r in manifest}
for source,out in exported:
 rel=str(out.relative_to(repo));assert rel not in known;known.add(rel)
 manifest.append({'path':rel,'source':str(source.relative_to(P.parents[1])),'sha256':sha(out),'bytes':out.stat().st_size})
put(repo/'catalog/source-manifest.json',manifest)
put(P/'LOCAL_EXPORT.json',{'repository':str(repo),'campaign':campaign,'accepted_rows_added':len(verified),'files':len(exported),'raw_included':False,'pushed':False,'website_changed':False})
readme=repo/'README.md'
text=readme.read_text();anchor='## Status snapshot'
assert anchor in text
section=f'## Quant/dequant correction (2026-11-10)\n\nThe dedicated server2 campaign covers all 71 fixed workloads with same-campaign eager controls. It adds {len(verified)-71} verified non-control variants, including explicitly separated Torch-expression compilation and compiled native antiquant wrappers. Exact cases, callable factories, source pins, and limitations are in [the campaign directory](reports/{campaign}/USAGE.json) and [report](reports/{campaign}/REPORT.json). Failed and unmeasured candidates remain separate in the catalog. These are device measurements; no raw data or website update is included. The general baseline retest remains paused.\n\n'
readme.write_text(text.replace(anchor,section+anchor,1))
print('Local export complete',len(verified),len(exported))
