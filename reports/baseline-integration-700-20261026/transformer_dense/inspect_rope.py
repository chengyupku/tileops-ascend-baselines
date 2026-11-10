import pathlib,json,hashlib
v=pathlib.Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/attention_vendor/src/ops-transformer/build/_CPack_Packages/Linux/External/cann-ops-transformer-classicpilot_attention_linux-aarch64.run/packages/vendors/classicpilot_attention_transformer')
a=[]
for f in v.rglob('*'):
 if f.is_file() and ('rotary_position_embedding' in str(f) or f.name=='libcust_opapi.so'):
  a.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
print(json.dumps({'vendor':str(v),'files':a},indent=2))
