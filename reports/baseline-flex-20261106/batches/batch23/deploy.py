import pathlib,json,hashlib,subprocess,shlex,time
P=pathlib.Path('/root/dyq/workspace/reports/baseline-flex-20261106');B=P/'batches/batch23';rows=json.load(open(B/'PLAN.json'))['rows'];audits={}
for h,host,root in [(1,'openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),(2,'openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace')]:
 d=P/f'server{h}'
 rel='TileOPs/benchmarks/ascend_all_pairs/vllm/linear_attention.py';r=subprocess.run(['ssh','-o','ClearAllForwardings=yes',host,'cat '+shlex.quote(root+'/'+rel)],capture_output=True,check=True);sha=hashlib.sha256(r.stdout).hexdigest();assert sha==hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest();audits[str(h)]={'provider_sha256':sha}
 while True:
  s=json.load(open(d/'STATE.json'))
  if s['phase']=='stopped' and not pathlib.Path('/proc/'+str(s['pid'])).exists():break
  time.sleep(3)
 print('DRAINED',h,flush=True)
 f=d/'measure_pair.py';old=f.read_bytes();(B/f'server{h}-measure.before.py').write_bytes(old);src=old.decode();needle="op in {'GDNFwdOp','DeltaNetFwdOp'}";assert needle in src;src=src.replace(needle,"op in {'GDNFwdOp','DeltaNetFwdOp','KDAFwdOp'}")
 needle='        original = registry.candidate_factory';assert needle in src
 insertion="""        if task.get('route') == 'original_vllm_kda':
            if op != 'KDAFwdOp' or library != 'vllm-ascend':
                raise ValueError('Invalid exact original KDA route')
            from benchmarks.ascend_all_pairs.vllm.linear_attention import linear_attention as original_kda
            def kda_factory(case, lib):
                plain = lambda v: json.loads(json.dumps(v))
                if lib != library or case.id != case_id or plain(case.params) != task['case']['params'] or plain(case._call.tensors) != task['case']['tensors']:
                    raise ValueError('KDA current exact contract mismatch')
                if any(case.params.get(k) for k in ('use_gate_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval')) or case.params['q'][0][1] > 16384:
                    raise ValueError('Unadmitted KDA resource/call-count scope')
                return lambda: original_kda(case)
            registry.candidate_factory = kda_factory
        original = registry.candidate_factory"""
 src=src.replace(needle,insertion);compile(src,str(f),'exec');f.write_text(src)
 remote=f'/opt/baseline-flex-20261106/server{h}/measure_pair.py';subprocess.run(['scp','-O','-q','-o','ClearAllForwardings=yes',str(f),host+':'+remote+'.next'],check=True)
 code='import pathlib,hashlib;p=pathlib.Path('+repr(remote+'.next')+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(hashlib.sha256(f.read_bytes()).hexdigest())+';p.replace('+repr(remote)+')';subprocess.run(['ssh','-o','ClearAllForwardings=yes',host,'python3 -c '+shlex.quote(code)],check=True)
 plan=d/'PLAN.json';old=plan.read_bytes();(B/f'server{h}-PLAN.before.json').write_bytes(old);data=json.loads(old);new=[t for t in rows if t['host']==f'server{h}'];assert not {t['id'] for t in new}&{t['id'] for t in data['rows']};data['rows']=new+data['rows'];tmp=plan.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(plan)
 (d/'STOP').unlink();subprocess.run(['systemd-run',f'--unit=baseline-flex-server{h}','--property=WorkingDirectory=/root/dyq/workspace','/usr/bin/python3',str(d/'controller.py')],check=True);print('RESUMED',h,len(new),flush=True)
(B/'DEPLOYED.json').write_text(json.dumps({'at':time.time(),'source_audits':audits,'added':len(rows),'original_shared_registry_unchanged':True},indent=2))
