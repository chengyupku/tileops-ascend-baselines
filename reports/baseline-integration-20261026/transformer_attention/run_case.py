from pathlib import Path
import os,sys,json,runpy,importlib.util,tempfile
P=Path(__file__).resolve().parent;W=P.parents[2];R=W/'TileOPs';cid=sys.argv[1];out=Path(sys.argv[2]).resolve()
assert ['ops-transformer','GQADenseFwdOp',cid] in [r['key'] for r in json.loads((P/'SCOPE.json').read_text())]
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert m['phase']=='running' and m['child']['pid']==os.getpgrp()
out.mkdir(exist_ok=False);os.chdir(out);short=Path('/proc')/str(os.getpid())/'cwd'
for name in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR']:
 (out/name).mkdir();os.environ[name]=str(short/name)
os.environ['TMPDIR']=str(short/'tmp');tempfile.tempdir=str(short/'tmp');sys.dont_write_bytecode=True
vendor=json.loads((P/'VENDOR.json').read_text())['installed_vendor'];os.environ['TRANSFORMER_ATTENTION_VENDOR']=vendor;os.environ['ASCEND_CUSTOM_OPP_PATH']=vendor;os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
sys.path[:0]=[str(R),str(R/'src'),str(W),str(P),str(W/'.cache/all-pairs/resume22_server2_transformer_attention01/native_build/runtime_deps/site-packages')]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
def frozen(name,file):
 spec=importlib.util.spec_from_file_location(name,P/file);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
frozen('benchmarks.ascend_graph_call','ascend_graph_call.py');mod=frozen('benchmarks.ascend_transformer_graph','ascend_transformer_graph.py')
from benchmarks.ascend_all_pairs import ops_transformer
ops_transformer.PROVIDERS['GQADenseFwdOp']=mod.transformer_graph_gqa
sys.argv=[str(R/'scripts/ascend_pair_check.py'),'--op','GQADenseFwdOp','--case-id',cid,'--library','ops-transformer','--output',str(out/'checker.json')]
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 import torch
 if torch.npu.is_initialized():torch.npu.synchronize()
 (out/'maps.txt').write_text(Path('/proc/self/maps').read_text())
