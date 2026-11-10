"""Foreground sequential jobs; every device child gets its own host guard."""
import pathlib,json,subprocess,os,sys,time
P=pathlib.Path(__file__).resolve().parent
start,end=map(int,sys.argv[1:3]);label=sys.argv[3] if len(sys.argv)>3 else 'v1'
failed=set();result=[]
for i in range(start,end):
 task=P/'tasks_v7'/f'{i:02d}.json';t=json.load(open(task));op=t['key'][1];out=P/'runs'/f'{i:02d}'/label;out.mkdir(parents=True,exist_ok=True)
 if op in failed:result.append({'ordinal':i,'status':'deferred_after_family_failure'});continue
 if (out/'measurement.json').exists():raise RuntimeError('refusing overwrite '+str(out))
 before=time.time()
 with (out/'prepare.log').open('w') as log:r=subprocess.run(['bash',str(P/'launch_v5.sh'),'prepare',str(task),str(out)],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:failed.add(op);print('FAILED PREPARE',i,flush=True);continue
 env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64')
 cmd=['python3','/opt/torch-v010-20261106/tools/npu_guard.py','run','--owner',f'root/torch66-server2/{i:02d}-{label}','--wait','3600','--timeout','900','--receipt',str(out/'guard.json'),'--','bash',str(P/'launch_v5.sh'),'device',str(task),str(out)]
 with (out/'device.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 row={'ordinal':i,'op':op,'exit':r.returncode,'seconds':time.time()-before}
 if r.returncode:failed.add(op)
 else:
  m=json.load(open(out/'measurement.json'));g=json.load(open(out/'guard.json'));assert g['performance_valid'] and g['workload_exit_code']==0
  row['us']=m['timing']['latency_ms']*1000;row['ratio']=row['us']/t['old_public']['tileops_latency_us'];row['baseline_gain']=t['old_public']['latency_us']/row['us']
 result.append(row);(P/f'batch-{start}-{end}-{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
