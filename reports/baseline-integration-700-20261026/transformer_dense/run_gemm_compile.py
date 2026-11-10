import subprocess,json,pathlib,time,datetime,os
p=pathlib.Path(__file__).resolve().parent/'gemm';(p/'guard_snapshot.json').write_text(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text());e=json.loads((p/'build_env.json').read_text());e['ASCEND_GLOBAL_LOG_LEVEL']='3';t=time.monotonic()
with (p/'compile33.log').open('w') as f:r=subprocess.run(json.loads((p/'build_command.json').read_text()),cwd=p/'run',env=e,stdout=f,stderr=subprocess.STDOUT)
(p/'compile33.json').write_text(json.dumps({'returncode':r.returncode,'elapsed_s':time.monotonic()-t,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));raise SystemExit(r.returncode)
