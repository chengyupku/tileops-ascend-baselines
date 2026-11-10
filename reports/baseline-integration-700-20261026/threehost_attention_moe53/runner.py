import sys,os,pathlib,json,runpy,hashlib,time,ctypes,tempfile
P=pathlib.Path(__file__).resolve().parent;F=P/'frozen/TileOPs';cid=sys.argv[1];O=P/cid;O.mkdir(exist_ok=True);os.chdir(O)
sys.path[:0]=[str(F),str(F/'src'),str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
for name in ['TMPDIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','ASCEND_PROCESS_LOG_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
 p=O/name;p.mkdir(exist_ok=True);os.environ[name]=str(p)
tempfile.tempdir=os.environ['TMPDIR']
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.ops_transformer import validated_runtime as vr
import dsa
native=[]
original=vr.load('ascend_transformer_graph');GC=original.GraphCall;old=GC.call

def traced(self,op,inputs,outputs,attrs,**kw):
 e={'op':op,'symbol':'aclopCompileAndExecute','pointer':ctypes.cast(self.execute,ctypes.c_void_p).value,'start':time.time(),'inputs':len(inputs),'outputs':len(outputs)};native.append(e)
 (O/'native.json').write_text(json.dumps(native,indent=2))
 r=old(self,op,inputs,outputs,attrs,**kw);self.torch.npu.synchronize();e['synchronized']=True;e['end']=time.time();return r
GC.call=traced
registry.candidate_factory=lambda case,library: lambda:dsa.factory(case)
sys.argv=[str(F/'scripts/ascend_pair_check.py'),'--op','DSADecodeWithKVCacheFwdOp','--case-id',cid,'--library','ops-transformer','--output',str(O/'checker.json')]
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 maps=pathlib.Path('/proc/self/maps').read_text();(O/'maps.txt').write_text(maps)
 files={}
 for line in maps.splitlines():
  p=pathlib.Path(line.split()[-1]);
  if p.is_absolute() and p.is_file() and '.so' in p.name:
   st=p.stat();files[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'inode':st.st_ino,'device':st.st_dev}
 (O/'native.json').write_text(json.dumps({'calls':native,'mapped_files':files},indent=2))
