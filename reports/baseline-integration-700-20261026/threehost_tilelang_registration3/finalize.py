import pathlib,json,hashlib
O=pathlib.Path(__file__).resolve().parent
p=json.loads((O/'PINS.json').read_text()); W='/opt/tileops-all-pairs/workspace'
p['runtime']={k:v for k,v in p['runtime'].items() if not '/reports/' in k and not '/tilelang-ascend/' in k}
p['runtime_provenance']={k:v for k,v in p['runtime_provenance'].items() if k in p['runtime']}
p['publication_ready']=False
p['remaining_evidence']=['Parent historical SDK/compiler closure audit pending; post-run inventory not substituted for historical pins.','Torch/torch_npu/TVM external runtime native dependencies appear in execution maps but complete historical content pins are not supplied in selected receipts.']
p['historical_native']=[]
E=O.parent/'threehost_tla_other54_r2/server1_evidence'
for r in json.loads((O/'SCOPE.json').read_text()):
 d=E/'runs'/('_'.join((r['baseline'],r['operator'],r['case_id'])))
 x=json.loads((d/'native_load_1.json').read_text());b=d/'native_artifacts'/x['sha256'];assert hashlib.sha256(b.read_bytes()).hexdigest()==x['sha256']
 p['historical_native'].append({'key':[r['baseline'],r['operator'],r['case_id']],'load_receipt':str(d/'native_load_1.json'),'binary_sha256':x['sha256'],'symbol':x['symbols'],'note':'historical evidence only; no old cache reuse'})
(O/'PINS.json').write_text(json.dumps(p,indent=2)+'\n')
s=(O/'registration.py').read_text()
s=s.replace("def environment():\n    pins=metadata()['PINS.json']","def environment():\n    pins=metadata()['PINS.json']")
s=s.replace("data=snapshot(runtime=runtime)","if not metadata()['PINS.json']['publication_ready']:\n        raise RuntimeError('Historical dependency closure audit pending; execution disabled')\n    data=snapshot(runtime=runtime)")
s=s.replace("Path(path).stat().st_ino!=int(fields[4])","(Path(path).stat().st_ino!=int(fields[4]) or (os.major(Path(path).stat().st_dev),os.minor(Path(path).stat().st_dev))!=tuple(int(x,16) for x in fields[3].split(':')))")
s=s.replace("if len(fields)!=6 or Path(fields[5].removesuffix(' (deleted)')).name not in names:continue", "if len(fields)!=6:continue\n        mapped=fields[5].removesuffix(' (deleted)')\n        relevant=Path(mapped).name in names or ('site-packages/tilelang/' in mapped and '.so' in mapped)\n        if not relevant:continue")
h={n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['CONTRACTS.json','PINS.json']}
s=s.replace('DATA_HASHES = {}','DATA_HASHES = '+repr(h))
(O/'registration.py').write_text(s)
