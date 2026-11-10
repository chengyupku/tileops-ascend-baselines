"""Pure CPU path/hash/ELF checks: no imports of runtime libraries."""
import json,hashlib,platform,struct,sys,datetime,os
from pathlib import Path
P=Path(__file__).resolve().parent
hosts=json.loads((P/'HOSTS.json').read_text())
host,pin=next((n,p) for n,p in hosts.items() if p['hostname']==platform.node())
bad=[];elf=[]
for name,h in pin['pins'].items():
 p=Path(name)
 if not p.is_file():bad.append({'path':name,'reason':'missing'});continue
 if hashlib.sha256(p.read_bytes()).hexdigest()!=h:bad.append({'path':name,'reason':'hash mismatch'})
 if p.name in ('bisheng','libtilelang_module.so','libtvm.so','libtvm_runtime.so'):
  b=p.read_bytes()[:20];elf.append({'path':name,'machine':int.from_bytes(b[18:20],'little') if b[:4]==b'\x7fELF' else None})
v=Path(pin['vendor'])
for p in [v/'src',v/'3rdparty/pto-isa/include',v/'3rdparty/tvm/python',Path(pin['base'])/'runtime/tilelang',Path(pin['base'])/'frozen/scripts/ascend_pair_check.py']:
 if not p.exists():bad.append({'path':str(p),'reason':'missing deployment path'})
import importlib.machinery
w=Path(pin['workspace']);c=Path(pin['compiler'])
paths=([str(c/'venv/lib/python3.12/site-packages'),str(w/'.venv/lib/python3.12/site-packages')] if host=='local' else [str(c/'python'),str(w/'.cache/all-pairs/remote_native_tla/python-deps'),str(w/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'),'/opt/tileops-ci/runtime/lib/python3.12/site-packages'])+sys.path
deps={n:(s.origin if (s:=importlib.machinery.PathFinder.find_spec(n,paths)) else None) for n in ['setuptools','torch','torch_npu','numpy','Cython','tvm_ffi']}
for n,v in deps.items():
 if v is None:bad.append({'dependency':n,'reason':'missing'})
expected_machine={'x86_64':62,'aarch64':183}.get(platform.machine())
for item in elf:
 if item['machine']!=expected_machine:bad.append({'path':item['path'],'reason':'ELF architecture mismatch'})
import subprocess
exports=[]
for item in elf:
 name=Path(item['path']).name
 if name not in ('libtvm.so','libtilelang_module.so'):continue
 proc=subprocess.run(['readelf','--dyn-syms','--wide',item['path']],capture_output=True,text=True)
 required='TVMFuncGetGlobal' if name=='libtvm.so' else 'CodeGenTileLangAscendPto'
 found=proc.returncode==0 and any(required in line and ' UND ' not in line for line in proc.stdout.splitlines())
 exports.append({'path':item['path'],'required_export_fragment':required,'found':found})
 if not found:bad.append({'path':item['path'],'reason':'missing compiler export'})
r=dict(exports=exports,dependencies=deps,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=host,python=sys.version,expected_python=pin['python'],architecture=platform.machine(),pin_count=len(pin['pins']),issues=bad,elf=elf,passed=not bad and sys.version==pin['python'],npu_runs=0)
print(json.dumps(r));(P/('PREFLIGHT_'+host+'.json')).write_text(json.dumps(r,indent=2)+'\n')
sys.exit(0 if r['passed'] else 1)
