import pathlib,json,hashlib,platform,subprocess,shutil,sys,datetime
P=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/opt/tileops-all-pairs/workspace');V=W/'ascend_baselines/tilelang-ascend'
R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'
C=W/'.cache/all-pairs/remote_tla_compiler/handoff'
sources=['examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py','examples/flash_attention/flash_attn_bhsd.py','examples/gqa_fwd_varlen/gqa_fwd_varlen.py','examples/causal_conv1d/causal_conv1d_pto.py','examples/normalization/rms_norm.py']
files={};missing=[]
for s in sources:
 src=V/s;dst=P/'original'/s;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
paths=[V/s for s in sources]+[P/'original'/s for s in sources]+[R/'tilelang/__init__.py',R/'tilelang/jit/adapter/cython/adapter.py',pathlib.Path('/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/bisheng'),W/'TileOPs/scripts/ascend_pair_check.py']+list((C/'lib').glob('*.so'))
for f in paths:
 if not f.exists():missing.append(str(f));continue
 b=f.read_bytes();files[str(f)]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),elf_machine=int.from_bytes(b[18:20],'little') if b[:4]==b'\x7fELF' else None,realpath=str(f.resolve()))
x=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=platform.node(),machine=platform.machine(),python=sys.version,files=files,missing=missing)
(P/'CPU_IDENTITY.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
