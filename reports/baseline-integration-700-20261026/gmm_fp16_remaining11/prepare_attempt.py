from pathlib import Path
import subprocess,sys,json
p=Path(__file__).resolve().parent
name,op,cid=sys.argv[1:];q=p/name;q.mkdir(exist_ok=False)
for n in ['capture.py','run.sh','prepare.py','submit.py']:
 s=(p/'conv02_fp32'/n).read_text().replace('gmm_fp16_remaining11/conv02_fp32','gmm_fp16_remaining11/'+name)
 s=s.replace('Conv1dFwdOp',op).replace('whisper-conv1-float16',cid)
 if n=='capture.py':s=s.replace('from conv1d_binding import conv1d as dense','from remaining_binding import remaining as dense')
 if n=='prepare.py':s=s.replace("P/'conv1d_binding.py'","P/'remaining_binding.py',P/'v5_fp32_bias.py'")
 (q/n).write_text(s)
for n in ['remaining_binding.py','v5_fp32_bias.py']:(q/n).write_text((p/n).read_text())
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';r='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11/'+name
files={f.name:f.read_text() for f in q.iterdir() if f.suffix in ('.py','.sh')}
code='from pathlib import Path\np=Path('+repr(r)+');p.mkdir(parents=True,exist_ok=True)\nfiles='+repr(files)+'\nfor n,t in files.items():(p/n).write_text(t)\n'
x=subprocess.run([ssh,'openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True);x.check_returncode()
x=subprocess.run([ssh,'openlibing-DevEnv-661495',f'python3 {r}/prepare.py'],capture_output=True,text=True);x.check_returncode()
print('Prepared',name,op,cid)
