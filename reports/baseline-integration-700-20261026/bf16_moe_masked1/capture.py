import os,sys,json,hashlib,importlib.util,time,shutil,ctypes,glob
from pathlib import Path
from datetime import datetime,timezone
P=Path(sys.argv[1]).resolve(); OP=sys.argv[2]; CID=sys.argv[3]; T=Path(__file__).resolve().parent; W=Path('/root/dyq/workspace')
def dump(n,x): (P/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# CPU pins before any device import
frozen=json.loads((T/'FROZEN.json').read_text())
parent=T.parent/'bf16_moe_hadamard_parent10'/'adapter.py'; orig=T.parent/'bf16_local_gemm18'/'binding.py'
assert sha(parent)==frozen['parent_adapter']; assert sha(orig)==frozen['orig_binding']
row=next(r for r in json.loads((T/'CASES.json').read_text()) if r['key']==[OP,'ds-v3-ep8-masked-bfloat16'] or r['key'][2]==CID)
assert row['key']==[OP,CID] or row['key'][2]==CID
# imports and device work begin only inside guarded command
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(T)]
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from adapter import make_candidate
from benchmarks.api import cases
from resource_gate import capacity
import torch, torch_npu
host=capacity(); free,total=torch.npu.mem_get_info(); disk=shutil.disk_usage(P).free; env=row['envelope']
ad=dict(host=host,hbm_free=free,hbm_total=total,disk_free=disk,**env,stage='before_input_allocation',case=CID,op=OP)
dump('RESOURCE_ADMISSION.json',ad)
if host['effective_available_host_bytes']<env['host_required'] or free<env['device_required'] or disk<env['disk_required']:
 dump('RESULT.json',{'status':'blocked','reason':'resource envelope unavailable','admission':ad}); raise SystemExit(78)
# exact case constructed from canonical manifest, unique coverage
matched=[c for c in cases(OP) if c.id==CID]; assert len(matched)==1
case=matched[0]; assert case.params==row['manifest_params'], (case.params,row['manifest_params'])
PROVIDERS[OP]=make_candidate
spec=importlib.util.spec_from_file_location('checker',W/'TileOPs/scripts/ascend_pair_check.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mod.require_guard()
sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op',OP,'--case-id',CID,'--library','ops-transformer','--output',str(P/'representative.json')]
t0=time.monotonic(); rc=mod.main(); elapsed=time.monotonic()-t0
torch.npu.synchronize()
# Actual live-process host library/symbol evidence, with addresses mapped to /proc/self/maps.
hostlib=T.parent/'bf16_local_gemm18/install/vendors/bf16gmm_transformer/op_api/lib/libcust_opapi.so'
cdll=ctypes.CDLL(str(hostlib)); syms={}
for name in ('aclnnGroupedMatmulV5GetWorkspaceSize','aclnnGroupedMatmulV5'):
 try:
  fn=getattr(cdll,name); addr=ctypes.cast(fn,ctypes.c_void_p).value
  syms[name]={'address':addr,'cdll_handle':int(cdll._handle)}
 except AttributeError: syms[name]={'missing':True,'cdll_handle':int(cdll._handle)}
maps=[]
for line in Path('/proc/self/maps').read_text().splitlines():
 if str(hostlib) in line:
  maps.append(line)
def addr_map(addr):
 if not addr:return None
 for line in maps:
  rng=line.split()[0]; a,b=(int(x,16) for x in rng.split('-'))
  if a<=addr<b:return line
 return None
for v in syms.values():
 if 'address' in v:v['maps_line']=addr_map(v['address'])
# capture only proof metadata; canonical checker owns full outputs/preservation.
dump('EXECUTION.json',{'utc':datetime.now(timezone.utc).isoformat(),'checker_exit_code':rc,'elapsed_seconds':elapsed,'phase':'reference_and_candidate_complete','coverage':1,'case':CID,'op':OP,'host_library':{'path':str(hostlib),'sha256':sha(hostlib),'cdll_handle':int(cdll._handle)},'symbols':syms,'host_library_maps':maps,'device_object_pin':frozen['device_object'],'native_source_pins':{'binding':sha(orig),'parent_adapter':sha(parent)},'actual_maps_observed_in_live_process':True})
dump('RESULT.json',{'status':'pass' if rc==0 else 'failed','checker_exit_code':rc,'checked_outputs':1 if rc==0 else 0,'total_outputs':1,'input_preservation_checked':rc==0,'coverage':1,'case':CID})
raise SystemExit(rc)
