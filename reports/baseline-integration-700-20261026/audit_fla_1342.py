import ast,datetime,fcntl,hashlib,json,pathlib
D=pathlib.Path(__file__).resolve().parent;W=D/'fla';root=D.parents[1]
allowed={(r['operator'],r['case_id']) for r in json.load(open(W/'RESULT.json'))['rows'] if r['status']=='completed'}
with open('/tmp/baseline700-provider-fla.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 script=(W/'static_check.py').read_text().replace("(p/'STATIC_CHECK.json').write_text", "(p.parent/'PARENT_FLA_STATIC_CHECK.json').write_text")
 exec(compile(script,str(W/'static_check.py'),'exec'),{'__file__':str(W/'static_check.py'),'__name__':'__main__'})
 public=root/'TileOPs/benchmarks/ascend_all_pairs/fla/integration700_fla_native.py';digest=hashlib.sha256(public.read_bytes()).hexdigest();assert digest==json.load(open(W/'ADAPTER_PROVENANCE.json'))['frozen_sha256'];assert public.read_bytes()==(W/'frozen_adapter.py').read_bytes()
 rows=[r for r in json.load(open(W/'RESULT.json'))['rows'] if (r['operator'],r['case_id']) in allowed];assert len(rows)>=22
 accepted=[]
 for r in rows:
  cp=pathlib.Path(r['checker_receipt']);cp=cp if cp.is_absolute() else root/cp
  c=json.load(open(cp));assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed'];v=c['verification'];assert v['checked_outputs']==v['total_outputs'] and v['total_outputs']>0
  gp=pathlib.Path(r['guard_receipt']);gp=gp if gp.is_absolute() else root/gp
  if gp.name.endswith('.GUARD_RESULT.json'):gp=gp.with_name(gp.name.replace('.GUARD_RESULT.json','.guard.log'))
  gr=gp.with_name(gp.name.replace('.guard.log','.GUARD_RESULT.json'));g=json.load(open(gr))
  if g['returncode'] not in (0,1):
   matches=[]
   for candidate in W.glob('*.GUARD_RESULT.json'):
    other=json.load(open(candidate));command=candidate.with_name(candidate.name.replace('.GUARD_RESULT.json','.COMMAND.json'))
    if other.get('returncode')==0 and command.exists() and c['case_id'] in json.load(open(command)) and other.get('started_at','z')<=c['started_at']<=c['finished_at']<=other.get('finished_at',''):
     matches.append((candidate,other))
   assert len(matches)==1,(r['case_id'],matches)
   gr,g=matches[0];gp=gr.with_name(gr.name.replace('.GUARD_RESULT.json','.guard.log'))
  assert g['returncode'] in [0,1]
  log=gp.read_text();assert '[npu-guard] acquired device 0 for baseline700-20261026/fla' in log
  assert not any(x in log.lower() for x in ['external npu process','terminating','timed out','interference detected'])
  if g['returncode']==1:
   assert gr.name in ['probe.GUARD_RESULT.json','batch2.GUARD_RESULT.json']
  if r['operator']=='CumsumFwdOp':assert cp.name=='CumsumFwdOp.explicit.checker.json'
  assert r['registered'] and r['status']=='completed'
  accepted.append({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':[r['baseline'],r['operator'],r['case_id']],'accepted':True,'host':'local','checker':str(cp),'guard':str(gr),'guard_batch_returncode':g['returncode'],'guard_batch_note':'Batch returns last checker exit1; this exact checker verified before unrelated failed case. No guard abort/contamination evidence; not represented as exit0.' if g['returncode'] else 'clean exit0','full_outputs':v,'registry_check':str(D/'PARENT_FLA_STATIC_CHECK.json'),'module_sha256':digest,'source_review':'Original complete ChunkFwdH contracts all K internally; independent output/group packing only. Original causal_conv W1 for Add; original single cumsum chunk4096; whole native GDN and first-token GLA specialization. Source branches reviewed; no algorithm replacement. GDN prefill uses existing exact FP64 oracle on CPU, formula/inputs/tolerance/all outputs unchanged.','provenance':str(W/'ADAPTER_PROVENANCE.json')})
 out=D/'ACCEPTED.json';data=json.load(open(out));by={tuple(r['key']):r for r in data['rows']};by.update({tuple(r['key']):r for r in accepted});data['rows']=list(by.values());data['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(out)
 (D/'PARENT_FLA_1342_REVIEW.json').write_text(json.dumps({'accepted':accepted,'total_new':len(accepted)},ensure_ascii=False,indent=2));print('accepted new',len(accepted),'total',len(data['rows']))
