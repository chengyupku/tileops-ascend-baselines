import pathlib,json,hashlib,base64,datetime,os
P=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_attention_moe53');v=json.loads((P/'frozen/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json').read_text());pins=v['groups']['attention']['pins'];errors=[]
for p,h in pins.items():
 try:
  if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h:errors.append(p)
 except OSError as e:errors.append([p,str(e)])
paths=[v['sources'][n] for n in ['ascend_graph_call','ascend_transformer_graph','ascend_transform_baselines']]+['/opt/tileops-all-pairs/control/private_python.py','/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libacl_op_compiler.so']
files={p:base64.b64encode(pathlib.Path(p).read_bytes()).decode() for p in paths}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checked=len(pins),errors=errors,files=files)))
