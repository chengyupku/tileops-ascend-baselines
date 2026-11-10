"""CPU-only early placement of large KDA/MLA receipt temporaries, without changing measurements."""
import subprocess,shlex,json
code='''import pathlib,json,shutil,os,time
root=pathlib.Path('/opt/torch-baseline400-20261104/server1/runs');spool=pathlib.Path('/dev/shm/torch400-receipt-spool');spool.mkdir(exist_ok=True);changes=[]
for profiles in root.glob('*/a*/profiles'):
 d=profiles.parent
 if (d/'DONE.json').exists() or (d/'measurement.json').exists() or os.path.lexists(d/'measurement.tmp'):continue
 task=json.loads((d/'task.json').read_text())
 if task['key'][1] not in ('KDAFwdOp','MLAVarlenFwdOp') or len(list(profiles.glob('repeat_*')))<8:continue
 size=sum(f.stat().st_size for f in profiles.rglob('*') if f.is_file())
 if size<2*1024**3 or shutil.disk_usage(spool).free<8*1024**3:continue
 target=spool/(d.parent.name+'-'+d.name+'-measurement.json')
 if target.exists():continue
 try:os.symlink(target,d/'measurement.tmp')
 except FileExistsError:continue
 row={'at':time.time(),'target':str(target),'reason':'large KDA/MLA complete raw retained; atomic final receipt uses availableSHM to preserve50GiB disk floor','cleanup':'only after local full archive SHA acceptance'}
 (d/'MEASUREMENT_SPOOL.json').write_text(json.dumps(row,indent=2));changes.append(row)
print(json.dumps(changes))'''
r=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495',shlex.join(['python3','-c',code])],text=True,capture_output=True,timeout=60,check=True)
print(r.stdout.strip())
