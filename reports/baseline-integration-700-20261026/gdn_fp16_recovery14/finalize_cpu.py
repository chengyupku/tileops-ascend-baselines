from pathlib import Path
import ast,json,hashlib,datetime,shutil,difflib
W=Path(__file__).resolve().parent;old=W.parent/'linear_deployment_recovery';P=W/'package';P.mkdir(exist_ok=True)
for n in ['binding.py','direct_acl.py','entry.py','RUNTIME_RECIPE.json','PINS.json']:shutil.copy2(old/'package'/n,P/n)
shutil.copy2(old/'ALL_CONTRACTS.json',W/'ALL_CONTRACTS.json')
f=P/'binding.py';s=f.read_text();patched=s.replace('else offsets_cpu.tolist()', 'else (offsets if offsets_cpu is None else offsets_cpu).tolist()');assert patched!=s;ast.parse(patched);f.write_text(patched)
(W/'metadata_optional_cpu.patch').write_text(''.join(difflib.unified_diff(s.splitlines(True),patched.splitlines(True),fromfile='a/package/binding.py',tofile='b/package/binding.py')))
class Values:
 def __init__(self,v):self.v=v
 def tolist(self):return self.v[:]
expr=ast.parse(patched);value=next(n.value for n in ast.walk(expr) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bounds' for t in n.targets));code=compile(ast.Expression(value),'<binding bounds>','eval');v=[0,1,64,1064,4096]
checks=[eval(code,{},dict(offsets=a,offsets_cpu=b)) for a,b in [(None,None),(Values(v),None),(Values(v),Values(v))]];assert checks==[None,v,v]
h=(W/'deployed_source/official_header.h').read_text();sig=h.split('aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(')[1].split(');')[0];args=[z.strip() for z in sig.split(',')];assert len(args)==30;assert all(x in h for x in ['gkOut','aqkOut','akkOut']);assert args[19].endswith('*gkOut') and args[20].endswith('*aqkOut') and args[21].endswith('*akkOut')
pro=json.loads((W/'DEPLOYED_PROVENANCE.json').read_text());rs={Path(r['path']).name:r for r in pro['files']};name='ChunkKdaFwd_2abd624f2831d861ffa7ac1b0049c12a';j=json.loads(rs[name+'.json']['text']);assert j['sha256']==rs[name+'.o']['sha256']
check=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata_results=checks,metadata_patch_only=True,abi_arguments=len(args),optional_outputs='gk FP32 [B,HV,S,K]; Aqk/Akk FP16 [B,HV,S,64]',matching_variant=name,binary_sha256=j['sha256'],binary_metadata_hash_match=True,source_build_attestation=False,npu_initialized=False)
(W/'CPU_CHECK.json').write_text(json.dumps(check,indent=2))
rows=json.loads((W/'SCOPE.json').read_text());rows=rows.get('rows',[]) if isinstance(rows,dict) else rows
for r in rows:
 r.update(status='blocked_provenance',reason='Current official deployment is present; exact opapi/tiling/binary-to-source build attribution prerequisite is not established. No new device run. Prior decode-b1 attention failure remains; other13 have no new numerical verdict.',checker_receipt=None,guard_receipt=None,api='aclnnChunkKdaFwd',registered=False,timings=dict(queue_seconds=None,device_seconds=None,compile_seconds=None));r.pop('evidence',None)
(W/'RESULT.json').write_text(json.dumps(dict(at=check['at'],rows=rows,new_npu_runs=0,verified_unregistered=0),indent=2))
(W/'HEARTBEAT.json').write_text(json.dumps(dict(at=check['at'],phase='CPU complete; blocked at required binary/source provenance prerequisite'),indent=2))
print(json.dumps(check,indent=2))
