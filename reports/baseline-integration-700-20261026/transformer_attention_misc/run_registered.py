import sys,os,json,runpy,time
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parents[2];R=W/'TileOPs';cid=sys.argv[1];op=sys.argv[2] if len(sys.argv)>2 else 'GQADenseFwdOp';lib=sys.argv[3] if len(sys.argv)>3 else 'ops-transformer';out=Path(os.environ['INTEGRATION700_RECEIPT_DIR'])
out.mkdir(parents=True,exist_ok=True);os.chdir(P)
import tempfile
for n in ["tmp","ASCEND_CACHE_PATH","ASCEND_WORK_PATH","ASCEND_PROCESS_LOG_PATH"]:
 (P/n).mkdir(exist_ok=True);os.environ["TMPDIR" if n=="tmp" else n]="/proc/self/cwd/"+n
tempfile.tempdir="/proc/self/cwd/tmp"
manifest=json.load(open(R/'benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json'))
paths=manifest['groups']['attention']['python_paths'];sys.path[:0]=[str(P),*paths[:-1]];sys.path.append(paths[-1])
(out/'started.json').write_text(json.dumps({'time':time.time(),'guard':json.load(open('/run/lock/ascend-npu/device-0.lock'))}))
sys.argv=[str(R/'scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library',lib,'--output',str(out/'checker.json')]
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 import torch
 if torch.npu.is_initialized():torch.npu.synchronize()
