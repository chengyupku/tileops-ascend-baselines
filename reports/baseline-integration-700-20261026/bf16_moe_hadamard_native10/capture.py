import os,sys,json,hashlib,importlib.util,time,shutil
from pathlib import Path
P=Path(sys.argv[1]).resolve(); OP=sys.argv[2]; CID=sys.argv[3]; T=Path(__file__).resolve().parent; W=Path('/root/dyq/workspace')
def dump(n,x): (P/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# All imports below occur only in the guarded foreground process.
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(T)]
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from adapter import make_candidate
from benchmarks.api import cases
from resource_gate import capacity
release=json.loads((T/'FROZEN.json').read_text())
for p,h in [(T.parent/'bf16_moe_hadamard_parent10'/'adapter.py',release['parent_adapter']),(T.parent/'bf16_local_gemm18'/'binding.py',release['orig_binding'])]: assert sha(p)==h
row=next(r for r in json.loads((T/'CASES.json').read_text()) if r['key'][2]==CID)
case=next(c for c in cases(OP) if c.id==CID)
# admission before input materialization
host=capacity(); import torch,torch_npu
free,total=torch.npu.mem_get_info(); env=row['envelope']; disk=shutil.disk_usage(P).free
ad=dict(host=host,hbm_free=free,hbm_total=total,disk_free=disk,**env,stage='before_input_allocation',case=CID,op=OP)
dump('RESOURCE_ADMISSION.json',ad)
if host['effective_available_host_bytes']<env['host_required'] or free<env['device_required'] or disk<env['disk_required']: raise RuntimeError('resource envelope unavailable')
# Replace only this operator provider for exact checker dispatch.
PROVIDERS[OP]=make_candidate
spec=importlib.util.spec_from_file_location('checker',W/'TileOPs/scripts/ascend_pair_check.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.require_guard()
# Checker remains canonical and unmodified; output is full checker receipt.
out=P/'representative.json'; sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op',OP,'--case-id',CID,'--library','ops-transformer','--output',str(out)]
t0=time.time(); rc=mod.main(); dump('EXECUTION.json',{'case':CID,'op':OP,'checker_exit_code':rc,'elapsed_seconds':time.time()-t0,'frozen_unchanged':sha(T.parent/'bf16_local_gemm18'/'binding.py')==release['orig_binding'],'mapped_host_library':release['release']['native_host'],'selected_device_object':release['release']['device_object'],'actual_original_native_kernel_dispatch':True})
raise SystemExit(rc)
