from pathlib import Path
import json,hashlib,datetime,collections,re,ast
P=Path(__file__).resolve().parent;W=P.parents[2];remote='/opt/tileops-all-pairs/workspace';data=json.loads((P/'RESULT.json').read_text());rows=data['rows'];success=[r for r in rows if r['status']=='verified_unregistered'];now=datetime.datetime.now(datetime.timezone.utc)
events=[]
for f in (P/'runs').glob('*/native.jsonl'):
 for line in f.read_text().splitlines():
  try:e=json.loads(line)
  except json.JSONDecodeError:continue
  events.append(e)
for n in ['REPS02_SGL_NATIVE.json','REPS03_SGL_NATIVE.json']:
 if (P/n).exists():events.extend(json.loads((P/n).read_text()))
kernels=set()
for f in (P/'source_evidence').rglob('*.py'):
 t=ast.parse(f.read_text())
 for x in t.body:
  if isinstance(x,ast.FunctionDef) and any('triton.jit' in ast.unparse(d) for d in x.decorator_list):kernels.add(x.name)
launches=[]
for e in events:
 text=e['text'];name=re.search(r'(?:kernel_name|registeredKernel)=([^,\s]+)',text);tm=re.search(r'(\d{4}-\d\d-\d\d)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',text)
 if name and tm and name[1] in kernels:
  launches.append(e|{'kernel':name[1],'timestamp':tm[1]+'T'+tm[2]+'.'+tm[3]+tm[4]+'+00:00'})
proof=[];missing=[]
for r in success:
 f=P/r['checker_receipt'];d=json.loads(f.read_text());start=d['started_at'];end=d['finished_at'];hits=[e for e in launches if start<=e['timestamp']<=end]
 # Keep precise representative native launches, deduplicated by original line.
 unique={ (x['file'],x['line']):x for x in hits};hits=list(unique.values())
 if not hits:missing.append(r['key'])
 proof.append({'key':r['key'],'checker':r['checker_receipt'],'window':[start,end],'native_launch_count':len(hits),'kernels':sorted({e['kernel'] for e in hits}),'events':hits[:12],'artifacts':str((f.parent/'NATIVE_ARTIFACTS.json').relative_to(P))})
 assert d['status']=='verified' and d['inputs_preserved'] and not d['oracle_source_changed'] and not d['provider_source_changed']
 assert d['verification']['checked_outputs']==d['verification']['total_outputs']>0
 for name,h in d.get('source_sha256',{}).items():
  candidate=Path(name.replace(remote,str(W))) if name.startswith(remote) else W/name
  if '/libraries/sgl-kernel-npu/' in name and not candidate.is_file():
   candidate=P/'source_evidence'/name.split('/libraries/sgl-kernel-npu/',1)[1]
  if candidate.is_file():assert hashlib.sha256(candidate.read_bytes()).hexdigest()==h,(candidate,h)
(P/'NATIVE_PROOF.json').write_text(json.dumps({'rows':proof,'missing_native_launch_keys':missing},indent=2))

print('verified receipts',len(success),'missing launch proof',missing)
