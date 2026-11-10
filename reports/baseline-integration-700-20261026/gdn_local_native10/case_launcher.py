"""Operational admission before unchanged canonical entry."""
import os,sys,json,runpy,datetime
from pathlib import Path
W=Path(__file__).resolve().parent
sys.path[:0]=[str(W/'chunk_package'),'/root/dyq/workspace/.venv/lib/python3.12/site-packages']
from direct_acl import require_guard
require_guard()
from resource_gate import require_capacity
run=Path(os.environ['GDN_RUN_DIR']);admission=require_capacity(W/'chunk_package',run)
import torch,torch_npu
torch.npu.set_device(0)
free,total=torch.npu.mem_get_info()
estimate=admission['required_envelope']['estimated_device_budget_bytes'];margin=2*1024**3
v={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'free_hbm_bytes':free,'total_hbm_bytes':total,'estimated_device_budget_bytes':estimate,'safety_margin_bytes':margin,'admitted':free>=estimate+margin}
(run/'HBM_ADMISSION.json').write_text(json.dumps(v,indent=2))
if not v['admitted']:raise RuntimeError('Insufficient actual free HBM with safety margin')
runpy.run_path(str(W/'chunk_package/entry.py'),run_name='__main__')
