from pathlib import Path
import json,shutil,hashlib
W=Path(__file__).resolve().parent;b=json.loads((W/'BLOCKERS.json').read_text());d=W/'diagnostics';d.mkdir(exist_ok=True)
for i in ['mamba2-2.7b-decode-b1-bfloat16','mamba2-780m-float16-float32']:
 for p in (W/i/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
  if p.parent.name=='plog' and p.parent.parent.name=='debug':shutil.copyfile(p,d/(i+'-'+p.name))
b['CausalConv1dUpdateFwdOp/qwen3-next-decode-b64-float16']={'reason':'Shared original ACLNN CausalConv1d package has host/JIT sources but no dynamic kernel configuration. Representative B1 native GetWorkspaceSize returned561002, ParseDynamicKernelConfig cannot find CausalConv1d JSON. No unchanged second-case attempt.','api':'aclnnCausalConv1dUpdate','evidence':['mamba2-2.7b-decode-b1-bfloat16/checker.json','diagnostics/']}
(W/'BLOCKERS.json').write_text(json.dumps(b,indent=2))
