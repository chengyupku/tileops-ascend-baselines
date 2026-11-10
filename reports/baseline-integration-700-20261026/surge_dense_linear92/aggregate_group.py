import pathlib,json,collections,datetime
P=pathlib.Path(__file__).resolve().parent
scope={};sources=collections.defaultdict(list);unknown=[]
def key(r):return tuple(r.get('key') or r.get('exact_key') or [r.get(k) for k in ['baseline','operator','case_id']])
for q in sorted(P.glob('worker*')):
 if not q.is_dir() or not (q/'SCOPE.json').exists():continue
 for r in json.loads((q/'SCOPE.json').read_text()):scope[key(r)]=r
 files=list(q.glob('*RESULT*.json'))+list(q.glob('RESULT.jsonl'))+list(q.glob('*/RESULT.json'))
 for f in files:
  try:
   a=[json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.suffix=='.jsonl' else json.loads(f.read_text())
   rr=a if isinstance(a,list) else a.get('rows',a.get('results',a.get('keys',[a])))
   for row_index,r in enumerate(rr):
    if not isinstance(r,dict):continue
    k=key(r)
    if k not in scope:continue
    sources[k].append({'file':str(f.relative_to(P)),'mtime':f.stat().st_mtime,'row_index':row_index,'row':r})
  except (ValueError,OSError,AttributeError):unknown.append(str(f.relative_to(P)))
rows=[]
for k,prior in scope.items():
 entries=sources.get(k,[])
 # Current replacement owns only three; original worker01 record not trusted for those.
 if k[0]=='flash-linear-attention-npu' and k[1]=='GDNFwdOp' and k[2] in ['q35-gva-decode-b8-float16','q35-gva-decode-b64-float16','q35-gva-decode-b512-float16']:
  entries=[e for e in entries if e['file'].startswith(('worker01r2/','worker01r3/'))]
 if entries:
  e=max(entries,key=lambda x:(x['mtime'],x['row_index']));r=dict(e['row']);r['key']=list(k);r['group_result_source']=e['file'];r['all_current_receipts']=sorted(set(x['file'] for x in entries))
 else:r={'key':list(k),'status':'pending_worker_delivery','prior_status':prior.get('status'),'group_result_source':None}
 if k==('flash-linear-attention-npu','GDNFwdOp','q35-gva-decode-b256-float16'):
  r={'key':list(k),'status':'prior_exact_numerical_failure_preserved','reason':prior['reason'],'checker':prior.get('checker'),'guard':prior.get('guard'),'group_result_source':'worker01/SCOPE.json','review_note':'Corrected worker01 stale representative-only claim using its supplied latest exact b256 scope receipt. No rerun.'}
 if k==('ops-nn','BatchNormFwdOp','resnet-stem-bfloat16'):
  r['checker']=prior.get('prior_checker');r['checked_outputs']=1;r['total_outputs']=3;r['review_note']='Original full checker coverage1/3, nonaccepted; do not infer full coverage from tensor-only result.'
 r['registered']=False;r['accepted_added']=0;rows.append(r)
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_count':len(scope),'delivered_keys':sum(r['status']!='pending_worker_delivery' for r in rows),'status_counts':dict(collections.Counter(r['status'] for r in rows)),'accepted_added':0,'parent_publishes_only':True,'rows':rows,'parse_issues':unknown}
t=P/'RESULT.tmp';t.write_text(json.dumps(out,ensure_ascii=False,indent=2));t.replace(P/'RESULT.json')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False))
