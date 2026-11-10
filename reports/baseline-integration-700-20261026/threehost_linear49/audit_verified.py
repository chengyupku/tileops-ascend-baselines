from pathlib import Path
import json,re,hashlib,os
P=Path(__file__).resolve().parent;scope={(x['baseline'],x['operator'],x['case_id']) for x in json.loads((P/'SCOPE.json').read_text())};audits=[]
for f in P.glob('**/checker.json'):
 if 'snapshot' in f.parts:continue
 c=json.loads(f.read_text());d=f.parent
 if c.get('status')!='verified':continue
 g=json.loads((d/'GUARD_RESULT.json').read_text());assert g['returncode']==0,(d,g['returncode'])
 key=(c['library'],c['op'],c['case_id']);assert key in scope
 assert c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 refs=c['observed_calls']['reference'];calls=c['observed_calls']['candidate'];nout=len(c['case_snapshot']['declared_output_names']);assert nout==2
 for x in refs+calls:assert x['returned'] and len(x['outputs']['items'])==nout
 maps=(d/'maps.txt').read_text();dispatch=[json.loads(x) for x in (d/'dispatch.jsonl').read_text().splitlines()];ptrchecks=[]
 for x in dispatch:
  ptr=int(x['symbol'],16);found=[]
  for line in maps.splitlines():
   fields=line.split();lo,hi=[int(v,16) for v in fields[0].split('-')]
   if lo<=ptr<hi:
    assert fields[-1]==x['library'];assert int(fields[4])==x['inode'];found.append(line)
  assert found;ptrchecks.append({'symbol':x['symbol'],'inode':x['inode'],'stat_device':x['device'],'maps_device':found[0].split()[3],'device_namespace_note':'stat reports overlay filesystem device; maps reports backing file device; both raw values retained, not asserted equal','map':found[0]})
 launches=[];bins=[]
 for log in (d/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
  for line in log.read_text(errors='replace').splitlines():
   if 'StreamLaunchKernel' in line and re.search(r'(ChunkGatedDeltaRule|RecurrentKda)_',line):launches.append(line)
   if 'Register Kernel succ' in line and re.search(r'(ChunkGatedDeltaRule|RecurrentKda)_',line):bins.append(line)
 assert launches and bins,d
 (d/'NATIVE_DISPATCH_EXTRACT.log').write_text('\n'.join(bins+launches)+'\n')
 a={'key':list(key),'checker_receipt':str(f.relative_to(P)),'guard_receipt':str((d/'GUARD_RESULT.json').relative_to(P)),'checker_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'guard_returncode':0,'inputs_preserved':True,'canonical_inputs':len(c['case_snapshot']['declared_input_names']),'canonical_outputs':nout,'reference_calls':len(refs),'candidate_calls':len(calls),'output_contract':calls[0]['outputs'],'pointer_maps_inode_device_verified':ptrchecks,'device_kernel_dispatch_lines':len(launches),'loaded_native_bin_lines':len(bins),'original_fullchecker_source_unchanged':True,'accepted_added':0,'source_hashes_receipt':str(f.relative_to(P))+'::source_sha256,oracle_source_sha256','runtime_hashes':str((d/'LOADED_HASHES.json').relative_to(P)),'native_pin_manifest':('LOCAL_DEPENDENCY_FREEZE.json' if dispatch[0]['library'].startswith('/root/') else 'server1/RECURRENT_NATIVE_FREEZE.json' if 'RecurrentKda' in c['api'] else 'server1/MIGRATED_NATIVE_FREEZE.json'),'complete_original_operator_sources':'COMPLETE_ORIGINAL_SOURCE_HASHES.json'}
 (d/'VALIDATION_AUDIT.json').write_text(json.dumps(a,indent=2));audits.append(a)
(P/'VERIFIED_AUDIT.json').write_text(json.dumps(audits,indent=2));print('verified audits',len(audits))
