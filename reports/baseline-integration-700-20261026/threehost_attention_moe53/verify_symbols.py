from pathlib import Path
import json,os
P=Path(__file__).resolve().parent;E=P/'remote_evidence';out=[]
for O in sorted(E.glob('mla-*')):
 if not (O/'checker.json').exists() or not (O/'native.json').exists():continue
 c=json.load(open(O/'checker.json'))
 if c.get('status')!='verified':continue
 n=json.load(open(O/'native.json'));maps=(O/'maps.txt').read_text().splitlines()
 for call in n['calls']:
  ptr=call['pointer'];found=[]
  for line in maps:
   f=line.split();lo,hi=(int(s,16) for s in f[0].split('-'))
   if lo<=ptr<hi:
    p=f[-1];pin=n['mapped_files'][p];assert pin['inode']==int(f[4]);assert 'libacl_op_compiler.so' in p
    found.append({'mapping':line,'file_pin':pin,'path':p,'maps_backing_device':f[3],'stat_mount_device':f'{os.major(pin["device"]):02x}:{os.minor(pin["device"]):02x}','device_note':'Maps backing device and path stat mount device differ in this container; both retained verbatim, inode and mapped path agree.'})
  assert len(found)==1 and call['synchronized'];out.append({'case_id':c['case_id'],'symbol':call['symbol'],'actual_pointer':ptr,'matching_maps':found,'device_api':call['op'],'synchronized':True})
(E/'ACTUAL_SYMBOL_ATTRIBUTION.json').write_text(json.dumps(out,indent=2));print('verified symbol mappings',len(out))
