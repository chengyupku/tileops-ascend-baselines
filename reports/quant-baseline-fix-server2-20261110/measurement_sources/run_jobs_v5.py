"""Foreground isolated jobs on server2; each device process has its own host guard."""
import json,subprocess,pathlib,sys,time,datetime
P=pathlib.Path(__file__).resolve().parent
HOST='openlibing-DevEnv-055409';REMOTE='/opt/quant-baseline-fix-server2-20261110'
jobs=json.loads(pathlib.Path(sys.argv[1]).read_text())
for job in jobs:
    idx=job['index'];variant=job['variant'];mode=job.get('mode','formal');attempt=job.get('attempt','a1')
    label=f'{mode}-{idx}-{variant}-{attempt}';lp=P/'jobs'/label;lp.mkdir(parents=True,exist_ok=True)
    if (lp/'exit.json').exists():continue
    if (P/'STOP').exists():break
    cmd=f'env LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64 /opt/tileops-ci/runtime/bin/python -I -S /opt/torch-v010-20261106/tools/npu_guard.py run --owner quant-baseline/{label} --wait 600 --timeout 600 --receipt {REMOTE}/{label}-guard.json -- bash {REMOTE}/launch_v5.sh {mode} {idx} {variant} {attempt}'
    print('START',label,flush=True)
    with (lp/'run.log').open('w') as f:
        result=subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','ServerAliveInterval=20',HOST,cmd],stdout=f,stderr=subprocess.STDOUT)
    (lp/'exit.json').write_text(json.dumps({'exit_code':result.returncode,'job':job,'finished':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    # Pull exact result + receipt; full raw collected separately, no remote deletion.
    fetch=f'cd {REMOTE} && tar -cf - runs/{idx}/{variant}/{attempt} {label}-guard.json'
    dest=P/'remote';dest.mkdir(exist_ok=True)
    with (lp/'fetch.log').open('w') as ferr:
        sender=subprocess.Popen(['ssh','-o','ClearAllForwardings=yes',HOST,fetch],stdout=subprocess.PIPE,stderr=ferr)
        receiver=subprocess.run(['tar','-C',str(dest),'-xf','-'],stdin=sender.stdout,stderr=ferr)
        sender.stdout.close();src=sender.wait()
    (lp/'fetch.json').write_text(json.dumps({'ssh_exit':src,'tar_exit':receiver.returncode}))
    print('DONE',label,result.returncode,flush=True)
    if result.returncode in (75,76,78,124,255):
        print('PAUSED: guard/infrastructure requires review',flush=True);break
