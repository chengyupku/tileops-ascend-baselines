import os,json,pathlib,platform,hashlib,datetime
root=pathlib.Path('/opt/tileops-all-pairs/workspace'); matches=[];apis=[]
for base in [root/'.cache/all-pairs',pathlib.Path('/usr/local/Ascend/cann/opp/vendors')]:
 for dp,ds,fs in os.walk(base):
  ds[:]=[d for d in ds if d not in {'.git','__pycache__','node_modules','triton','logs','kernel_meta'}]
  for f in fs:
   p=pathlib.Path(dp)/f
   if f in ('libcust_opapi.so','libcust_opsproto.so','libcust_opmaster_rt2.0.so','libopapi.so') and ('transformer' in str(p) or 'vendors' in str(p)):
    apis.append({'path':str(p),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
   if f in ('rotary_position_embedding.py','grouped_matmul.py','bias_gate_gelu.py','moe_init_routing_v2.py','moe_finalize_routing_v2.py','norm_rope_concat.py','fused_linear_online_max_sum.py','fused_add_topk_div.py','trans_quant_param.py') and ('transformer' in str(p) or 'vendors' in str(p)):
    matches.append(str(p))
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'machine':platform.machine(),'host_api':apis,'wrappers':matches,'source_root_exists':(root/'ascend_baselines/ops-transformer').exists()},indent=2))
