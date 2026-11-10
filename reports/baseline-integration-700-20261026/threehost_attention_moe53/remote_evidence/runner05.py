import sys,os,pathlib,json,runpy,hashlib,time,ctypes,tempfile
P=pathlib.Path(__file__).resolve().parent;F=P/'frozen/TileOPs';cid=sys.argv[1];O=P/(cid+'-attempt05');O.mkdir(exist_ok=True);os.chdir(O)
sys.path[:0]=[str(F),str(F/'src'),str(P)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
for name in ['TMPDIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','ASCEND_PROCESS_LOG_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
 p=O/name;p.mkdir(exist_ok=True);os.environ[name]=str(p)
os.environ['TMPDIR']=f'/proc/{os.getpid()}/cwd/TMPDIR'
tempfile.tempdir=os.environ['TMPDIR']
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.ops_transformer import validated_runtime as vr
import dsa
native=[]
original=vr.load('ascend_transformer_graph');GC=original.GraphCall;old=GC.call

def traced(self,op,inputs,outputs,attrs,**kw):
 e={'op':op,'symbol':'aclopCompileAndExecute','pointer':ctypes.cast(self.execute,ctypes.c_void_p).value,'start':time.time(),'inputs':len(inputs),'outputs':len(outputs)};native.append(e)
 (O/'native.json').write_text(json.dumps(native,indent=2))
 try:
  r=old(self,op,inputs,outputs,attrs,**kw)
 except BaseException:
  f=self.base.aclGetRecentErrMsg;f.restype=ctypes.c_char_p
  e['acl_recent_error']=(f() or b'').decode(errors='replace');e['end']=time.time();raise
 self.torch.npu.synchronize();e['synchronized']=True;e['end']=time.time();return r
GC.call=traced
registry.candidate_factory=lambda case,library: lambda:dsa.factory(case)
sys.argv=[str(F/'scripts/ascend_pair_check.py'),'--op','DSADecodeWithKVCacheFwdOp','--case-id',cid,'--library','ops-transformer','--output',str(O/'checker.json')]
def stage_profile(frame,event,arg):
 if event in ('call','return') and frame.f_code.co_name in ('gen_inputs','ref_program','factory') and frame.f_code.co_filename.endswith('/dsa.py'):
  with (O/'stages.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'event':event,'function':frame.f_code.co_name,'file':frame.f_code.co_filename})+'\n')
sys.setprofile(stage_profile)
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 sys.setprofile(None)
 maps=pathlib.Path('/proc/self/maps').read_text();(O/'maps.txt').write_text(maps)
 files={}
 for line in maps.splitlines():
  p=pathlib.Path(line.split()[-1]);
  if p.is_absolute() and p.is_file() and '.so' in p.name:
   st=p.stat();files[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'inode':st.st_ino,'device':st.st_dev}
 (O/'native.json').write_text(json.dumps({'calls':native,'mapped_files':files},indent=2))
