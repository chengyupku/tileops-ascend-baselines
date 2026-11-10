import pathlib,subprocess,tarfile,io,sys,json
P=pathlib.Path(__file__).resolve().parent;host,run=sys.argv[1:]
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
rp='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker05/runs/'+run
code="import pathlib,tarfile,sys; p=pathlib.Path("+repr(rp)+"); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|'); [t.add(x,arcname=x.name) for x in p.iterdir() if x.is_file() or x.name.startswith('launch_')]; t.close()"
import shlex
r=subprocess.run([ssh,host,'python3 -c '+shlex.quote(code)],capture_output=True);out=P/('server1_runs' if host.endswith('661495') else 'server2_runs')/run;out.mkdir(parents=True,exist_ok=True)
(out/'harvest.stderr').write_bytes(r.stderr)
if r.returncode:raise SystemExit(r.returncode)
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(out,filter='data')
print(str(out),len(r.stdout))
