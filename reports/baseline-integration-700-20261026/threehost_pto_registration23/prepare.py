import ast,json,pathlib,hashlib,shutil,datetime
P=pathlib.Path(__file__).resolve().parent; S=P.parent/'threehost_pto_original23'
def save(n,d): (P/n).write_text(json.dumps(d,indent=2)+'\n')
save('HEARTBEAT.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='building_registration',exact_keys=[r['key'] for r in json.loads((S/'RESULT.json').read_text())['rows']],last_output='Original factories located in frozen benchmark package; local public modules absent',blocker=None,next='freeze adapter and bind original host environments'))
(P/'closure/private_tla').mkdir(parents=True,exist_ok=True)
for n in ['adapter.py','private_tla/native.py','private_tla/deployment.py','private_tla/__init__.py']:
 shutil.copyfile(S/'server1_evidence'/n,P/'closure'/n)
shutil.copyfile(S/'server1_evidence/frozen/benchmarks/api.py',P/'closure/api.py')
# Generic CPU metadata and kernel-lock validation only; no DeltaNet runtime or contract reused.
tree=ast.parse((P.parent/'threehost_linear_registration8/registration.py').read_text())
names={'plain','canonical','contract','key','_proc','require_guard','validate_inputs','checked','implementation'}
parts=[ast.get_source_segment((P.parent/'threehost_linear_registration8/registration.py').read_text(),n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
header='import ast,copy,hashlib,json,os,sys,types,platform\nfrom pathlib import Path\nDATA_ROOT=Path(__file__).resolve().parent\n'
(P/'registration.py').write_text(header+'\n\n'.join(parts).replace('GDN_CASE_OWNER','PTO23_OWNER')+'\n')
rows=json.loads((S/'RESULT.json').read_text())['rows']
proposals={tuple(r['key']):r for r in json.loads((S/'REGISTRATION_PROPOSAL.json').read_text())['rows']}
save('EVIDENCE.json',[dict(key=r['key'],host=r['host'],checker=r['checker'],guard=r['guard'],seed=r['input_seed'],existing_factory=proposals[tuple(r['key'])]['existing_factory'],prestart_link=str(S/'PRESTART_LINK.json'),artifacts=r['artifacts']) for r in rows])
pins={};hosts={}
for host in ['local','server1','server2']:
 e=S/(host+'_evidence'); f=json.loads((e/'HOST_FREEZE.json').read_text())
 local=host=='local';w=pathlib.Path('/root/dyq/workspace' if local else '/opt/tileops-all-pairs/workspace')
 base=w/'reports/baseline-integration-700-20261026/threehost_pto_original23';base=base/'local_evidence' if local else base
 v=w/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/tla_pristine_build/source' if local else w/'ascend_baselines/tilelang-ascend'
 c=v.parent if local else w/'.cache/all-pairs/remote_tla_compiler'
 sdk=pathlib.Path('/usr/local/Ascend/cann-9.2.0-beta.2') if local else w/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
 selected={p:r['sha256'] for p,r in f['files'].items() if '/runtime/' in p or '/original/' in p or p.endswith('/bisheng') or pathlib.Path(p).name in ['libtilelang_module.so','libtvm.so','libtvm_runtime.so'] or '/pto-isa/include/' in p or str(v/'src')+'/' in p}
 hosts[host]=dict(hostname=f['host'],python=f['python'],base=str(base),vendor=str(v),compiler=str(c),sdk=str(sdk),workspace=str(w),pins=selected)
 # Preserve original setup statements verbatim, wrapped with supplied paths/out.
 src=(e/'run_case_fresh.py').read_text() if (e/'run_case_fresh.py').exists() else (S/'run_case_fresh.py').read_text()
 start=src.index("for n in ['ASCEND_NPU_ARCH'");end=src.index('import setuptools')
 setup=src[start:end]
 (P/('setup_'+host+'.py')).write_text('from pathlib import Path\nimport os,sys,tempfile\ndef setup(P,W,V,C,R,out):\n'+''.join(' '+line+'\n' for line in setup.splitlines()))
save('HOSTS.json',hosts)
save('LOCAL_PINS.json',{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'closure').rglob('*') if p.is_file()})
# Verify pre-existing archive hashes, without dumping table.
expected=json.loads((S/'FILEHASHES.json').read_text())['files'];bad=[];count=0
for n,h in expected.items():
 p=S/n
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append(n)
 count+=1
save('SOURCE_HASH_CHECK.json',dict(checked=count,mismatches=bad))
