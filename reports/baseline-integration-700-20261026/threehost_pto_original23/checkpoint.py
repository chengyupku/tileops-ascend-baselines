from pathlib import Path
import json,hashlib,struct,time
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text());historical=json.loads((P.parent/'threehost_pto_historical23/REQUALIFICATION.json').read_text())['rows'];old={tuple(x['key']):x for x in historical}
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
rows=[]
for s in scope:
 key=(s['baseline'],s['operator'],s['case_id']);r=dict(key=list(key),status='not_yet_run',accepted_added=0);tag='_'.join(key)
 attempts=[]
 for guard in sorted(P.glob(tag+'*.guard.json'),key=lambda f:json.loads(f.read_text())['finished']):
  g=json.loads(guard.read_text());suffix=guard.name[len(tag):-len('.guard.json')];d=P/(g['host']+'_evidence')/'runs'/(key[1]+'_'+key[2]+suffix)
  a=dict(guard=str(guard),guard_exit=g['returncode'],run=str(d));attempts.append(a)
  if suffix=='_local':
   a['excluded_reason']='Initial local host freeze did not finish before process startup; subsequent complete evidence run required'
   continue
  if not (d/'checker.json').exists():continue
  c=json.loads((d/'checker.json').read_text());a.update(checker_status=c['status'],error=c.get('error'))
  if c['status']!='verified' or g['returncode']!=0:continue
  assert (c['library'],c['op'],c['case_id'])==key
  assert c['inputs_preserved'] and c['verification']['checked_outputs']==c['verification']['total_outputs']==1
  h=old[key]['historical_checker'];assert c['input_seed']==h['input_seed'] and c['case_snapshot']==h['case_snapshot']
  dispatch=list(d.glob('*.dispatch.json'));assert dispatch
  artifacts=[]
  for f in dispatch:
   t=json.loads(f.read_text());assert t['completed'] and t['target']=='pto' and t['platform']=='A5'
   sym=t['symbols']['call'];assert sym['dladdr_rc']==1 and sym['pointer']==sym['symbol_address'] and sym['filename']==t['path']
   mappings=[l.split() for l in t['loaded']['maps'].splitlines() if len(l.split())>5 and l.split()[-1]==t['path']]
   assert mappings and all(int(m[4])==t['inode'] for m in mappings)
   assert any(int(m[0].split('-')[0],16)<=sym['pointer']<int(m[0].split('-')[1],16) and 'x' in m[1] for m in mappings)
   lib=d/'native_artifacts'/t['sha256'];assert sha(lib)==t['sha256'];data=lib.read_bytes();assert data[:4]==b'\x7fELF'
   shoff=struct.unpack_from('<Q',data,40)[0];entsize,count,stridx=struct.unpack_from('<HHH',data,58)
   headers=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*entsize) for i in range(count)];sh=headers[stridx];names=data[sh[4]:sh[4]+sh[5]];sections=[]
   for sh in headers:
    name=names[sh[0]:].split(b'\x00',1)[0].decode()
    if 'aicore' in name.lower() and sh[1]!=8:
     dst=lib.parent/(lib.name+'.'+name.replace('.','_'));dst.write_bytes(data[sh[4]:sh[4]+sh[5]]);sections.append(dict(section=name,sha256=sha(dst),path=str(dst),size=sh[5]))
   assert any(x['section']=='.aicore_binary' for x in sections)
   build=json.loads((P/(g['host']+'_evidence')/'provenance/0_build_receipt.json').read_text());expected={x['path']:x['sha256'] for x in build['artifacts']}
   loaded=[x for x in t['loaded']['libraries'] if 'libtvm.so' in x['path'] or 'libtilelang_module.so' in x['path']];assert len(loaded)==2 and all(expected[x['path']]==x['sha256'] for x in loaded)
   orig=json.loads((P/'original_hashes.json').read_text());libgen=[x for x in t['loaded']['python_modules'] if x['path'].endswith('/tilelang/jit/adapter/libgen.py')];assert len(libgen)==1 and libgen[0]['sha256']==orig['tilelang/jit/adapter/libgen.py']
   artifacts.append(dict(dispatch=str(f),elf=str(lib),elf_sha256=sha(lib),device_sections=sections,loaded_compiler=loaded,loaded_libgen=libgen))
  r.update(status='full_checker_native_verified',host=g['host'],checker=str(d/'checker.json'),guard=str(guard),verification=c['verification'],inputs_preserved=True,input_seed=c['input_seed'],original_case_seed_unchanged=True,artifacts=artifacts,registered=False)
 
 if r['status']=='not_yet_run' and attempts:
  r['status']='guard_not_started' if all(a['guard_exit']==75 for a in attempts) else 'attempt_incomplete_or_failed'
 r['attempts']=attempts;rows.append(r)
result=dict(at=time.time(),accepted_added=0,total=23,full_checker_native_verified=sum(r['status']=='full_checker_native_verified' for r in rows),rows=rows)
(P/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
