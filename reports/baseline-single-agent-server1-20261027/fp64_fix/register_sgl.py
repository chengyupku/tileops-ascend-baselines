from pathlib import Path
import json,hashlib,fcntl,os,subprocess,tarfile,io,datetime
P=Path(__file__).resolve().parent;W=P.parents[2];D=W/'reports/baseline-integration-700-20261026'
rows=[];contracts={}
for f in sorted((P/'runs/sgl-kernel-npu').glob('*/checker.json')):
 d=json.loads(f.read_text());assert d['status']=='verified' and d['inputs_preserved'] and not d['oracle_source_changed'] and not d['provider_source_changed']
 assert d['verification']['checked_outputs']==d['verification']['total_outputs']==2
 assert not json.loads((f.parent/'SOURCE_AFTER.json').read_text())['changed']
 g=P/('sgl-kernel-npu--'+d['case_id']+'.guard.json');assert json.loads(g.read_text())['exit_code']==0
 contracts[d['case_id']]={'parameters':d['case_snapshot']['parameters'],'declared_tensors':d['case_snapshot']['declared_tensors']}
 rows.append((f,g,d))
assert len(rows)==6
config=P/'SGL_VERIFIED_CONTRACTS.json';config.write_text(json.dumps(contracts,indent=2)+'\n')
sha=hashlib.sha256((P/'sgl_adapter.py').read_bytes()).hexdigest();chash=hashlib.sha256(config.read_bytes()).hexdigest()
module='''"""Six exact server1 GDN contracts, verified with original FP64 CPU reference."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]/'reports/baseline-single-agent-server1-20261027/fp64_fix'
PINS=PIN_VALUE
for name,digest in PINS.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Verified GDN binding changed: '+name)
CONTRACTS=json.loads((ROOT/'SGL_VERIFIED_CONTRACTS.json').read_text())
def install(providers):
    previous=providers.get('GDNFwdOp')
    def exact(case):
        if case.id not in CONTRACTS:
            return previous(case)
        plain=lambda v:json.loads(json.dumps(v))
        c=CONTRACTS[case.id]
        if plain(case.params)!=c['parameters'] or plain(case._call.tensors)!=c['declared_tensors']:
            raise ValueError('Verified GDN exact contract changed')
        guard=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
        if guard.get('phase')!='running' or guard.get('child',{}).get('pid')!=os.getpgrp():
            raise RuntimeError('Full foreground NPU guard required')
        for p in ['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(ROOT.parents[2]/'.cache/all-pairs/remote_native_tla/python-deps')]:
            if p not in sys.path:sys.path.append(p)
        spec=importlib.util.spec_from_file_location('_single_server1_sgl_exact',ROOT/'sgl_adapter.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        return adapter.build(case)
    exact.verified_exact_case_ids=tuple(CONTRACTS)
    providers['GDNFwdOp']=exact
'''.replace('PIN_VALUE',repr({'sgl_adapter.py':sha,'SGL_VERIFIED_CONTRACTS.json':chash}))
f=W/'TileOPs/benchmarks/ascend_all_pairs/sgl/single_server1_fp64.py';f.write_text(module)
init=f.with_name('__init__.py');append='\nfrom .single_server1_fp64 import install as _install_single_server1_fp64\n_install_single_server1_fp64(PROVIDERS)\n'
with open('/tmp/baseline700-provider-sgl.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 s=init.read_text()
 if '_install_single_server1_fp64(PROVIDERS)' not in s:init.write_text(s+append)
# Remote additive install uses the existing live init, preserving unrelated registrations.
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w') as t:
 for p in [f,config]:t.add(p,arcname=str(p.relative_to(W)))
ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');host='openlibing-DevEnv-661495'
subprocess.run([ssh,host,'tar -xf - -C /opt/tileops-all-pairs/workspace'],input=b.getvalue(),check=True)
code=f'''from pathlib import Path
import fcntl
p=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py')
with open('/tmp/baseline700-provider-sgl.lock','a') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 s=p.read_text()
 if '_install_single_server1_fp64(PROVIDERS)' not in s:p.write_text(s+{append!r})
'''
subprocess.run([ssh,host,'python3 -'],input=code,text=True,check=True)
# CPU-only registration smoke: lazy imports, exact ownership and fallback.
code=f'''import sys,importlib.util,types
from pathlib import Path
p=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/sgl/single_server1_fp64.py')
spec=importlib.util.spec_from_file_location('check_sgl_registration',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r={{'GDNFwdOp':lambda c:'fallback'}};m.install(r)
assert len(r['GDNFwdOp'].verified_exact_case_ids)==6
assert r['GDNFwdOp'](types.SimpleNamespace(id='not-owned'))=='fallback'
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
print('CPU lazy registry/fallback: passed, six exact keys')
'''
r=subprocess.run([ssh,host,'python3 -'],input=code,text=True,capture_output=True,check=True);print(r.stdout)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();accepted=[]
for checker,guard,d in rows:
 accepted.append({'at':now,'key':[d['library'],d['op'],d['case_id']],'accepted':True,'host':host,'checker':str(checker),'checker_sha256':hashlib.sha256(checker.read_bytes()).hexdigest(),'guard':str(guard),'registration':str(f),'source_after':str(checker.parent/'SOURCE_AFTER.json'),'native_trace':str(checker.parent/'PHASES.jsonl'),'verification':d['verification'],'reference_device':'cpu','validator_device':'cpu','reference_formula_and_tolerances_unchanged':True,'performance_claim':False,'note':'Same original factory registered without replaying successful case; CPU registry smoke verified exact ownership and fallback. User authorized FP64 device dispatch correction.'})
(P/'SGL_ACCEPTED_ADDITIONS.json').write_text(json.dumps({'rows':accepted},indent=2)+'\n')
for ledger in [P.parent/'ACCEPTED.json',D/'ACCEPTED.json']:
 with open(ledger.with_name('ACCEPTED.lock'),'a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads(ledger.read_text());existing={tuple(x['key']) for x in d['rows']};d['rows'] += [x for x in accepted if tuple(x['key']) not in existing];d['updated_at']=now;tmp=ledger.with_suffix('.json.sgl.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(ledger)
print('registered and accepted',len(accepted))
