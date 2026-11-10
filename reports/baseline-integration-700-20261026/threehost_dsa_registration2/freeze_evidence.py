from pathlib import Path
import json,hashlib,shutil,datetime
O=Path(__file__).resolve().parent;E=O.parent/'threehost_attention_moe53/remote_evidence';F=E/'frozen/TileOPs';pins={}
def copy(p,rel):
 d=O/'evidence'/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);pins[str(d.relative_to(O))]=hashlib.sha256(d.read_bytes()).hexdigest()
for n in ['ACTUAL_SYMBOL_ATTRIBUTION.json','DSA96_DISPATCH_AND_BINARIES.json','DSA128_DISPATCH_AND_BINARIES.json','DSA_BINDING_DIFF.txt']:copy(E/n,n)
for case in ['mla-d576-h128-kv96k-bfloat16-attempt06','mla-d576-h128-kv128k-bfloat16-attempt01']:
 for n in ['checker.json','native.json','guard.json','receipt.json','command.json','maps.txt']:copy(E/case/n,case+'/'+n)
 c=json.loads((E/case/'checker.json').read_text());assert hashlib.sha256((F/'scripts/ascend_pair_check.py').read_bytes()).hexdigest()==c['verifier_sha256']
 for kind in ['oracle_source_sha256','provider_source_sha256']:
  for path,digest in c[kind].items():
   p=E/'frozen'/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,path;copy(p,'frozen/'+path)
for n in ['DSA96_DISPATCH_AND_BINARIES.json','DSA128_DISPATCH_AND_BINARIES.json']:
 for path,item in json.loads((E/n).read_text())['artifacts'].items():
  p=E/path.split('/threehost_attention_moe53/',1)[1];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];copy(p,'device_binary/'+p.name)
copy(F/'scripts/ascend_pair_check.py','frozen/TileOPs/scripts/ascend_pair_check.py')
assert (E/'dsa.original.py').read_bytes()==(F/'benchmarks/ascend_all_pairs/ops_transformer/integration700_transformer_attention_misc_dsa.py').read_bytes()
(O/'EVIDENCE_PINS.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pins=pins,verified_original_provider_and_oracle_bytes=True,unchanged_fullchecker=True),indent=2)+'\n')
print('frozen',len(pins),'exactcase/checker/reference/provider/device artifacts')
