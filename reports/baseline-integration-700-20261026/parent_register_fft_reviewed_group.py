import pathlib,json,sys,hashlib,fcntl,os,types,datetime,sqlite3,shutil
sys.dont_write_bytecode=True
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';name=sys.argv[1];assert name in ['fft_remaining_c128_2','fft_remaining_c64_7'];P=D/name;N=W/'TileOPs/benchmarks/ascend_all_pairs/ops_math';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 t=p.with_name(p.name+'.parent.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True);assert db.execute('select status from jobs where id=?',('17b45d1f18b64176',)).fetchone()[0]=='completed','Peer checker/finalize still active'
r=read(P/'PARENT_REVIEW.json');assert r['status']=='eligible_pending_registration';rows=read(P/'RESULT.json')['rows'];keys=[tuple(x) for x in r['keys']];deps={}
for row in rows:
 c=read(pathlib.Path(row['checker']));deps.update(c['source_sha256'])
if (P/'EXACTKEYS.json').exists():
 assert sha(P/'EXACTKEYS.json')==read(P/'FROZEN.json')['EXACTKEYS.json'];deps[str((P/'EXACTKEYS.json').relative_to(W))]=sha(P/'EXACTKEYS.json')
for p,h in deps.items():assert sha(W/p)==h,p
put(P/'REGISTER_DEPENDENCIES.json',deps)
module=(D/'fft_window_first1/integration700_fft_window_first1.py').read_text();module=module.replace(str(D/'fft_window_first1'),str(P)).replace("_KEY = ('FFTC2CFwdOp', 'latency-4k-complex64')",'_KEYS = frozenset('+repr([k[1:] for k in keys])+')').replace('!=_KEY','not in _KEYS').replace('fft_window_first1_frozen',name+'_frozen')
old_adapter=sha(D/'fft_window_first1/adapter.py');old_dep=sha(D/'fft_window_first1/REGISTER_DEPENDENCIES.json');module=module.replace(old_adapter,sha(P/'adapter.py')).replace(old_dep,sha(P/'REGISTER_DEPENDENCIES.json'))
prefix='_'+name;modname='integration700_'+name
append=f'''\n# Original complete aclStft receipts: {name}; system CANN AiCPU.
from . import {modname} as {prefix}
{prefix}_prior = PROVIDERS.get('FFTC2CFwdOp')
def {prefix}_route(case, _prior={prefix}_prior):
    if (case._call.signature.name, case.id) in {prefix}._KEYS:
        return {prefix}.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact FFT route')
PROVIDERS['FFTC2CFwdOp'] = {prefix}_route
'''
sys.path.insert(0,str(W/'TileOPs'))
with open('/tmp/baseline700-provider-ops_math.lock','a') as l:
 fcntl.flock(l,fcntl.LOCK_EX);old=(N/'__init__.py').read_text();assert prefix+'_route' not in old;compile(module,modname,'exec');compile(old+append,'registry','exec');(P/'PARENT_PROVIDER_BEFORE.py').write_text(old);(P/(modname+'.py')).write_text(module);(N/(modname+'.py')).write_text(module);(N/'__init__.py').write_text(old+append)
 try:
  from benchmarks.ascend_all_pairs import registry,ops_math
  m=getattr(ops_math,prefix);assert sha(pathlib.Path(m._load().__file__))==sha(P/'adapter.py');m._load=lambda:types.SimpleNamespace(make=lambda c:('expected',c.id));os.environ['ASCEND_CLASSIC_MATH_VENDOR']=m._VENDOR;os.environ['ASCEND_CUSTOM_OPP_PATH']=m._VENDOR
  for _,op,cid in keys:
   case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)));assert registry.candidate_factory(case,'ops-math')()==('expected',cid)
  case.id='__unknown__';assert getattr(ops_math,prefix+'_route')(case,_prior=lambda c:'prior')=='prior'
  first=ops_math._fft_window_first1;first._load=lambda:types.SimpleNamespace(make=lambda c:('first',c.id));case.id='latency-4k-complex64';assert registry.candidate_factory(case,'ops-math')()==('first',case.id)
  assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 except BaseException:
  (N/'__init__.py').write_text(old);raise
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(str(D/'ACCEPTED.json')+'.lock','a') as l:
 fcntl.flock(l,fcntl.LOCK_EX);a=read(D/'ACCEPTED.json');before=len(a['rows']);assert not {tuple(x['key']) for x in a['rows']}&set(keys);shutil.copy2(D/'ACCEPTED.json',D/('ACCEPTED.before_'+name+'_'+now.replace(':','-')+'.json'))
 for row in rows:a['rows'].append({'key':[row['baseline'],row['operator'],row['case_id']],'accepted':True,'at':now,'host':'local','checker':row['checker'],'guard':row.get('guard',row.get('guard_receipt')),'parent_review':str(P/'PARENT_REVIEW.json'),'qualification':r['qualification']})
 a['updated_at']=now;put(D/'ACCEPTED.json',a);total=len(a['rows'])
r.update(status='accepted',accepted_added=len(keys),accepted_total=total,accepted_at=now,registration_wait=None,formal_registry_exact_keys=len(keys),module_sha256=sha(N/(modname+'.py')),registry_sha256=sha(N/'__init__.py'));put(P/'PARENT_REVIEW.json',r)
x=read(D/'PLAN.json')
for w in x['workers']:
 if w['name']==name:w.update(active_assignment=False,ownership='accepted',completed_parent_accepted=len(keys))
put(D/'PLAN.json',x);s=read(D/'STATE.json');pending=any(e['directory']==str(P) for e in s['pending_registration_scopes']);s['pending_registration_scopes']=[e for e in s['pending_registration_scopes'] if e['directory']!=str(P)];s['ownership_audit']['pending_registration' if pending else 'active_execution']-=len(keys);s['ownership_audit']['accepted']=total;s.update(completed=total,parent_accepted_completed=total,last_status_accepted=total,last_status_at=now,active_workers=0,last_acceptance_review=str(P/'PARENT_REVIEW.json'));s['closed_success_scopes'].append(name);put(D/'STATE.json',s)
M=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(M)+'.lock','a') as l,open(M.parent/'STATE.lock','a') as b:
 fcntl.flock(l,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(M,M.with_name('STATE.before_accept_'+name+'_'+now.replace(':','-')+'.json'));mem=read(M);mem['integration_700_20261026'].update({k:s[k] for k in ['parent_accepted_completed','last_status_at','active_workers','last_acceptance_review','ownership_audit','pending_registration_scopes','closed_success_scopes']});put(M,mem)
print({'group':name,'added':len(keys),'accepted':total,'remaining':700-total})
