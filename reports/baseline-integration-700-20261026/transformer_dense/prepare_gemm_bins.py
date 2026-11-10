from pathlib import Path
import shutil,subprocess,json
p=Path(__file__).parent/'gemm';v=p/'private_provider';root=v/'op_impl/ai_core/tbe/kernel';dst=root/'ascend950/grouped_matmul_apt';dst.mkdir(parents=True,exist_ok=True)
for column in ['22','33','fp32','fp8','fp8half']:
 for f in (p/'package'/column).glob('*'):
  if f.suffix in ('.o','.json'):shutil.copy2(f,dst/f.name)
config=root/'config/ascend950';config.mkdir(parents=True,exist_ok=True)
cmd=['/opt/tileops-ci/runtime/bin/python','/usr/local/Ascend/cann-9.1.0/aarch64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_ops_config.py','-p',str(root/'ascend950'),'-s','ascend950','-o',str(config)]
r=subprocess.run(cmd,capture_output=True,text=True);(p/'CONFIG_BUILD.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2));print(r.returncode,r.stdout,r.stderr)
