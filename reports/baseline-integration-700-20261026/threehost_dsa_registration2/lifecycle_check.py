# Pure metadata fixtures for runtime lifetime; no native imports or native factory calls.
import sys,types,json,datetime
from pathlib import Path
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O.parent/'threehost_attention_moe53/remote_evidence/frozen/TileOPs'))
import registration as M
from benchmarks.ascend_baselines import Candidate
class NoCopy:
 def __deepcopy__(self,memo):raise AssertionError('Cached device state copied')
c=types.SimpleNamespace(inputs=NoCopy(),workload=NoCopy(),params={},id='fixture',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='DSADecodeWithKVCacheFwdOp')))
data=M.snapshot();guard=['g'];counter={k:0 for k in ['make','run','sync','cleanup']};admissions=[]
M.matches=lambda c:True;M.contract=lambda c:{};M.validate_inputs=lambda *a:None
M.require_guard=lambda:guard[0];M.environment=lambda:'e';M.loaded_libraries=lambda:None
M.synchronize=lambda:counter.__setitem__('sync',counter['sync']+1)
def admit(runtime=True):admissions.append(runtime);return data,guard[0],'e'
M.admit=admit
def factory(case):
 counter['make']+=1
 def run(*a):counter['run']+=1;return 'fixture'
 def cleanup():counter['cleanup']+=1
 return types.SimpleNamespace(implementation=types.SimpleNamespace(run=run),cleanup=cleanup)
M.private_load=lambda d:lambda p:types.SimpleNamespace(factory=factory)
a=M.make(c);assert counter['make']==0
assert a.implementation.run()=='fixture' and a.implementation.run()=='fixture'
assert admissions==[True,False] and counter==dict(make=1,run=2,sync=2,cleanup=0)
guard[0]='other'
for f in [a.implementation.run,a.cleanup]:
 try:f()
 except RuntimeError:pass
 else:raise AssertionError('Guard crossing accepted')
guard[0]='g';a.cleanup();a.cleanup();assert counter['cleanup']==1
try:a.implementation.run()
except RuntimeError:pass
else:raise AssertionError('Closed candidate reused')
(O/'LIFECYCLE_CHECK.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',admissions=admissions,counters=counter,guard_crossing_rejected=True,closed_reuse_rejected=True,fixture_only=True),indent=2)+'\n')
print('CPU metadata fixtures: runtime hash once; repeated run sync; guard crossing rejected; cleanup exactly once')
