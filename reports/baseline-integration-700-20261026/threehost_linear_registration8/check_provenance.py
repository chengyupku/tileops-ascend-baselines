import pathlib,json,hashlib,tarfile,struct,datetime
O=pathlib.Path(__file__).resolve().parent;P=O.parent/'threehost_linear49';pins=json.loads((O/'RUNTIME_PINS.json').read_text());rows=[]
def sha(raw):return hashlib.sha256(raw).hexdigest()
for cid,x in pins.items():
 paths={x['recipe']['opapi'],*x['registered_device_objects']};remote=x['host']['architecture']=='aarch64';checked=[]
 if remote:
  archive=P/('recurrent_custom92_aarch64.tar.gz' if cid.startswith('decode-') else 'native_custom92_aarch64.tar.gz')
  with tarfile.open(archive) as t:
   for name in sorted(paths):
    member='kdapairs_transformer/'+name.split('/kdapairs_transformer/',1)[1];raw=t.extractfile(member).read()
    assert sha(raw)==x['runtime'][name];assert raw[:4]==b'\x7fELF'
    checked.append({'path':name,'sha256':sha(raw),'elf_machine':struct.unpack('<H',raw[18:20])[0],'archive':archive.name})
 else:
  for name in sorted(paths):
   raw=pathlib.Path(name).read_bytes();assert sha(raw)==x['runtime'][name];assert raw[:4]==b'\x7fELF'
   checked.append({'path':name,'sha256':sha(raw),'elf_machine':struct.unpack('<H',raw[18:20])[0]})
 assert next(v for v in checked if v['path']==x['recipe']['opapi'])['elf_machine']==(183 if remote else 62)
 rows.append({'case_id':cid,'native_api_and_actual_registered_objects':checked})
source=[]
for n in ('ORIGINAL_SOURCE_PINS.json','RECURRENT_ORIGINAL_SOURCE_PINS.json'):
 for row in json.loads((P/n).read_text()):
  if '/source_templates/' not in row['path']:continue
  suffix=row['path'].split('/attention/',1)[1];f=P/'original_source'/suffix
  assert sha(f.read_bytes())==row['sha256']
  source.append({'file':str(f.relative_to(P)),'sha256':row['sha256'],'earlier_pins':n})
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'cases':rows,'original_api_source_links':source,'limits':'Only the listed earlier-pinned API/source files are execution-chain anchors. Later collected complete source tree is supplemental review material, not pre-run attestation. ARM native archives are complete original custom packages; not the abandoned runtime92 partial archive.'}
(O/'PROVENANCE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(rows),'earlier_source_links':len(source)}))
