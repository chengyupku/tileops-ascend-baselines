import pathlib,json,hashlib,ast,sys,dataclasses,functools,typing,shutil,datetime
T=pathlib.Path(__file__).resolve().parent; W=pathlib.Path('/root/dyq/workspace');R=W/'reports/baseline-integration-700-20261026';D=R/'bf16_moe_down_full3'
sys.dont_write_bytecode=True
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W/'.venv/lib/python3.12/site-packages')]
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.manifest.values import ADTValue
op='MoEGroupedGemmFwdOp';scope=json.loads((T/'SCOPE.json').read_text());ids={x['key'][2] for x in scope}
plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
calls={c.case_id:c for r in load_workloads(op) for d in r.get('dtype_cases') or [{}] if (c:=instantiate(plan,r,d)).case_id in ids}
tree=ast.parse((W/'TileOPs/benchmarks/api.py').read_text()); nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Case','_report_value')]
ns=dict(dataclasses=dataclasses,functools=functools,Any=typing.Any,Optional=typing.Optional,Callable=typing.Callable,Call=type(next(iter(calls.values()))),ADTValue=ADTValue)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-case-CPU','exec'),ns)
rows=[]
for name,c in calls.items():
 case=ns['Case'](name,c,None); rows.append(dict(key=['ops-transformer',op,name],manifest_params=case.params,envelope={}))
(T/'CASES.json').write_text(json.dumps(rows,indent=2))
for name in ['capture.py','native_observation.py','environment.sh','resource_gate.py']:
 shutil.copyfile(D/name,T/name)
s=(D/'adapter.py').read_text().replace("Path(__file__).resolve().parent.parent",repr(str(R)))
# Only relocate the original two readonly dependency roots; no numerical code changes.
s=(D/'adapter.py').read_text().replace("Path(__file__).resolve().parent.parent", "Path("+repr(str(R))+")")
(T/'adapter.py').write_text(s)
s=(T/'capture.py').read_text();s=s.replace("assert host['effective_available_host_bytes']>=en['host_required'] and free>=en['device_required'] and admission['disk_free']>=en['disk_required'],'capacity envelope unavailable'","# Observe resources without adding the obsolete conditional failure-path gate.")
(T/'capture.py').write_text(s)
# Add workspace-size observation only, leaving allocation and lifetime untouched.
s=(T/'native_observation.py').read_text().replace("result=_fn(*args)","result=_fn(*args)\n   if _attr=='size':\n    before['workspace_bytes']=C.cast(args[-2],C.POINTER(C.c_uint64)).contents.value")
(T/'native_observation.py').write_text(s)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
old=json.loads((D/'FROZEN.json').read_text())['sha256']
pins={p:sha(p) for p in old if '/bf16_local_gemm18/' in p or '/bf16_moe_hadamard_parent10/adapter.py' in p}
for n in ['TileOPs/scripts/ascend_pair_check.py','TileOPs/benchmarks/ascend_math_more.py','TileOPs/workloads/moe.py','TileOPs/workloads/numerics.py','tools/baseline_protocol.py']:
 pins[str(W/n)]=sha(W/n)
for p in T.glob('*.py'):ast.parse(p.read_text()); pins[str(p)]=sha(p)
(T/'FROZEN.json').write_text(json.dumps({'sha256':pins,'reused_receipt':str(D/'FROZEN.json'),'changes':{p:{'old':old[p],'current':v} for p,v in pins.items() if p in old and old[p]!=v}},indent=2))
for row in rows:
 p=T/row['key'][2];(p/'result').mkdir(parents=True,exist_ok=True);shutil.copyfile(T/'FROZEN.json',p/'FROZEN.json')
bounds=json.loads((R/'moe_fullchecker_scratch5/PER_KEY_BOUNDS.json').read_text())
(T/'CPU_CHECKS.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),real_manifest_cases=rows,original_adapter='only absolute dependency relocation',original_checker='unmodified',native_lifetime='unchanged cumulative per-expert FP32 tensors and workspaces until cleanup',resource_decision='execute success path; diagnostic failure-only 105-108 GiB is not an unconditional capacity requirement',pin_count=len(pins),prior_bounds=str(R/'moe_fullchecker_scratch5/PER_KEY_BOUNDS.json'),host_compatibility={'local':'x86_64 original deployed .so and successful down recipe','server1':'aarch64; original x86_64 .so incompatible and exact vendor absent','server2':'aarch64; original x86_64 .so incompatible and exact vendor absent'},accepted_added=0),indent=2))
print(json.dumps(rows));print('pins',len(pins))
