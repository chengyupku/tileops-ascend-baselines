import json,sys,pathlib,datetime
payload=json.load(sys.stdin);out=[]
for x in payload:
 p=pathlib.Path(x['path'])
 try:d=json.load(open(p))
 except Exception as e:out.append(dict(x,error=str(e)));continue
 if x.get('kind')=='result_verified':
  matching=[r for r in d.get('rows',[]) if isinstance(r,dict) and r.get('key')==x['key']]
  for r in matching:
   ck=r.get('checker_receipt') or r.get('checker_path') or r.get('checker');ck=ck.get('path') if isinstance(ck,dict) else ck
   info=dict(x,row_summary={k:v for k,v in r.items() if k in ['status','api','new_execution','guard_exit_code','guard_exit','inputs_preserved','original_source_call_build_evidence','source_pins','shared_family_ref','checker_receipt','guard_receipt']})
   if ck:
    try:c=json.load(open(ck));info.update(checker=ck,summary={k:v for k,v in c.items() if k in ['op','library','case_id','started_at','finished_at','status','verification','inputs_preserved','api','verifier_sha256','provider_source_changed','oracle_source_changed']})
    except Exception as e:info['checker_error']=str(e)
   out.append(info)
 else:out.append(dict(x,checker=str(p),summary={k:v for k,v in d.items() if k in ['op','library','case_id','started_at','finished_at','status','verification','inputs_preserved','api','verifier_sha256','provider_source_changed','oracle_source_changed']}))
print(json.dumps(out,ensure_ascii=False))
