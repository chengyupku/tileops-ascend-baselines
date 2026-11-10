from pathlib import Path
import json,hashlib,ast,subprocess,time
T=Path(__file__).resolve().parent;R=T.parent;D=R/'bf16_local_gemm18';W=Path('/root/dyq/workspace')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(T/n).write_text(json.dumps(x,indent=2)+'\n')
a=(R/'bf16_bmm_conv_grouped_prep10/adapter.py').read_text().replace("p.get('T') != 'bfloat16'","p.get('T') != 'float16'").replace('This preparation is explicitly BF16 only','This exact scope is explicitly FP16 only').replace('dtype=t.bfloat16','dtype=getattr(t, p[\'T\'])')
a+='''\n\ndef make_candidate(case):
    import json
    key=('ops-transformer',case._call.signature.name,case.id)
    task=Path(__file__).resolve().parent
    rows=json.loads((task/'CASES.json').read_text())
    if key not in [tuple(r['key']) for r in rows]:
        raise ValueError('Outside exact FP16 scope')
    path=task.parent/'bf16_local_gemm18/binding.py'
    pins=json.loads((task/'FROZEN.json').read_text())['sha256']
    return candidate(case,native_factory=load_factory(path,sha256=pins[str(path)]))
'''
(T/'adapter.py').write_text(a)
for name in ['native_observation.py','resource_gate.py']:(T/name).write_bytes((R/'bf16_moe_masked_recovery1'/name).read_bytes())
c=(R/'bf16_moe_masked_recovery1/capture.py').read_text().replace('P.mkdir(exist_ok=True)','P.mkdir(parents=True,exist_ok=True)').replace(";OP=row['key'][1]",";assert OP==row['key'][1]")
start=c.index("  if label=='candidate':");end=c.index('  for j,t in enumerate(args):save',start);c=c[:start]+c[end:]
c=c.replace("store='output' in name","store=True")
(T/'capture.py').write_text(c)
rows=[]
for op,cid,p in [('GroupedGemmFwdOp','moe-dgrad-float16',dict(M=4096,G=16,N=4096,K=4096,transpose_a=False,transpose_b=False,T='float16')),('BmmFwdOp','square-2k-float16',dict(B=8,M=2048,N=2048,K=2048,T='float16')),('GroupedGemmFwdOp','moe-fwd-float16',dict(M=4096,G=16,N=4096,K=4096,transpose_a=False,transpose_b=True,T='float16'))]:
    A=p.get('B',1)*p['M']*p['K'];B=p.get('B',p.get('G',1))*p['K']*p['N'];Y=p.get('B',1)*p['M']*p['N'];E=A+B+Y
    # Includes full inputs/weights, FP32 operands retained until cleanup, native outputs,
    # native workspace allowance, checker copies/temporary masks, full CPU reference,
    # capture CPU contiguous/byte copies; distinct device and host budgets.
    en=dict(input_elements=A+B,output_elements=Y,host_required=E*48+4*2**30,device_required=E*24+4*2**30,disk_required=E*16+2**30,workspace_allowance=4*2**30)
    rows.append(dict(key=['ops-transformer',op,cid],manifest_params=p,envelope=en));(T/cid).mkdir(exist_ok=True)
dump('CASES.json',rows)
old=json.loads((D/'square-1k-bfloat16/FROZEN.json').read_text())['sha256'];pins={p:h for p,h in old.items() if '/install/' in p or '/site-packages/' in p or '/TileOPs/' in p or p.endswith('/tools/baseline_protocol.py') or p.endswith('/binding.py') or p=='/usr/local/bin/npu-guard'};pins.update(json.loads((D/'SOURCE_PINS.json').read_text()))
changed={p:dict(old=h,current=sha(p)) for p,h in pins.items() if sha(p)!=h}
assert all(p==str(W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py') for p in changed),changed
for p in changed:pins[p]=sha(p)
sdk=Path('/usr/local/Ascend/cann-9.2.0-beta.2')
for p in [sdk/'set_env.sh']+list((sdk/'x86_64-linux/lib64').glob('*.so*')):
    if p.is_file():pins[str(p)]=sha(p)
for name in ['DEPLOYMENT_MANIFEST.json','SOURCE_PINS.json','ORIGINAL_BOUNDARY.json','DEVICE_COMMAND.json']:pins[str(D/name)]=sha(D/name)
for p in T.glob('*.py'):ast.parse(p.read_text())
assert 'OP=sys.argv[2];CASE_ID=sys.argv[3]' in c
assert '/root/dyq/workspace/.venv/bin/python -B "$P/capture.py" "$P/$2/result" "$1" "$2"' in (T/'launch_one.sh').read_text()
subprocess.run(['bash','-n',str(T/'launch_one.sh'),str(T/'environment.sh')],check=True)
for p in list(T.glob('*.py'))+[T/'CASES.json',T/'launch_one.sh',T/'environment.sh']:pins[str(p)]=sha(p)
dump('FROZEN.json',dict(sha256=pins,shared_registry_updates=changed))
history=json.loads((R/'gmm_fp16_remaining11/RESULT.json').read_text())['results'];dump('HISTORY_REVIEW.json',dict(rows=[r for r in history if r['key'][2] in [x['key'][2] for x in rows]],conclusion='Remote ARM Torch2.10 timeouts; dgrad unrun. Local x86 Torch2.13 differs; no identical local canonical found in specified evidence. BF16 representative failure is not FP16 numerical evidence.',remote_environments=[json.loads((R/'fp16_reference_cpu4'/f).read_text().splitlines()[0]) for f in ['openlibing-DevEnv-055409.log','openlibing-DevEnv-661495.log']]))
dump('CPU_CHECKS.json',dict(ast='pass',launch_arguments='pass',python='absolute .venv under guard',pins=len(pins),shared_registry_updates=changed,resource_envelopes=rows))
dump('HEARTBEAT.json',dict(time=time.time(),stage='prepared'))
print('prepared',len(pins))
