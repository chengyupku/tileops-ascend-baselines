import pathlib,json,hashlib,subprocess,shlex,time
P=pathlib.Path('/root/dyq/workspace/reports/baseline-flex-20261106');D=P/'server1';B=P/'batches/batch25';rows=json.load(open(B/'PLAN.json'))['rows'];host='openlibing-DevEnv-661495'
while True:
 s=json.load(open(D/'STATE.json'))
 if s['phase']=='stopped' and not pathlib.Path('/proc/'+str(s['pid'])).exists():break
 time.sleep(3)
f=D/'measure_pair.py';old=f.read_bytes();(B/'measure.before.py').write_bytes(old);src=old.decode();needle="(op in {'GDNFwdOp','DeltaNetFwdOp','KDAFwdOp'} and library == 'vllm-ascend')";assert needle in src;src=src.replace(needle,"(op in {'GDNFwdOp','DeltaNetFwdOp','KDAFwdOp'} and library == 'vllm-ascend') or (op == 'GDNFwdOp' and library == 'sgl-kernel-npu')")
needle='        original = registry.candidate_factory';assert needle in src
new="""        if task.get('route') == 'original_sgl_gdn_fp16':
            if op != 'GDNFwdOp' or library != 'sgl-kernel-npu':
                raise ValueError('Invalid original SGL FP16 GDN route')
            from benchmarks.ascend_all_pairs.sgl.linear import gdn as original_sgl_gdn
            def sgl_gdn_factory(case, lib):
                plain = lambda v: json.loads(json.dumps(v))
                p = case.params
                if lib != library or case.id != case_id or plain(p) != task['case']['params'] or plain(case._call.tensors) != task['case']['tensors']:
                    raise ValueError('SGL GDN current exact contract mismatch')
                if p['q'][1] != 'float16' or p['q'][0][1] > 8192 or p['q'][0][2:] != p['v'][0][2:] or any(p.get(k) for k in ('use_gate_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval')):
                    raise ValueError('Unadmitted SGL FP16 scope')
                if os.environ.get('GDN_USE_MEGA_GDN','0') != '0':
                    raise ValueError('Require original default SGL complete pipeline')
                return lambda: original_sgl_gdn(case)
            registry.candidate_factory = sgl_gdn_factory
        original = registry.candidate_factory"""
src=src.replace(needle,new);compile(src,str(f),'exec');f.write_text(src)
remote='/opt/baseline-flex-20261106/server1/measure_pair.py';subprocess.run(['scp','-O','-q','-o','ClearAllForwardings=yes',str(f),host+':'+remote+'.next'],check=True);code='import pathlib,hashlib;p=pathlib.Path('+repr(remote+'.next')+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(hashlib.sha256(f.read_bytes()).hexdigest())+';p.replace('+repr(remote)+')';subprocess.run(['ssh','-o','ClearAllForwardings=yes',host,'python3 -c '+shlex.quote(code)],check=True)
f=D/'PLAN.json';old=f.read_bytes();(B/'PLAN.before.json').write_bytes(old);d=json.loads(old);assert not {t['id'] for t in rows}&{t['id'] for t in d['rows']};d['rows']=rows+d['rows'];tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(f)
(D/'STOP').unlink();subprocess.run(['systemd-run','--unit=baseline-flex-server1','--property=WorkingDirectory=/root/dyq/workspace','/usr/bin/python3',str(D/'controller.py')],check=True);(B/'DEPLOYED.json').write_text(json.dumps({'at':time.time(),'added':len(rows),'original_source_and_shared_registry_unchanged':True},indent=2));print('DEPLOYED18',flush=True)
