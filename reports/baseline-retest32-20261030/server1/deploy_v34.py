from pathlib import Path
import json,time,base64,subprocess
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';s=(r/'tools/baseline_retest32_server1_entry_v33.py').read_text();mark='sys.argv=args';i=s.index(mark);s=s[:i]+'''if task['key'][:2]==['flash-linear-attention-npu','GDNFwdOp'] and task['case']['params']['q'][1]=='float16':
    from benchmarks.ascend_all_pairs import registry
    from benchmarks.ascend_baselines import source_hashes
    import dataclasses
    original_factory=registry.candidate_factory
    def gdn_factory(case,library):
        make=original_factory(case,library)
        if library!='flash-linear-attention-npu' or case._call.signature.name!='GDNFwdOp':return make
        def build():
            candidate=make()
            original_run=candidate.implementation.run
            def run(q,k,v,g,beta,*rest,**kwargs):
                return original_run(q,k,v,g.float(),beta.float(),*rest,**kwargs)
            return dataclasses.replace(candidate,implementation=dataclasses.replace(candidate.implementation,run=run),source_sha256={**candidate.source_sha256,**source_hashes(Path(__file__))},adaptation=candidate.adaptation+'; per-call lossless FP16 gate/beta to FP32 for original Phase6 ABI')
        return build
    registry.candidate_factory=gdn_factory
'''+s[i:];files={'tools/baseline_retest32_server1_entry_v34.py':s,'tools/baseline_retest32_server1_python_v34.sh':(r/'tools/baseline_retest32_server1_python_v33.sh').read_text().replace('entry_v33','entry_v34')}
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v34.py').write_text(s);s=s.replace("t['id'] not in ('be0", "t['id']!='7c7db643272857967545',t['id'] not in ('be0",1).replace("return (x['key'][0]=='DeepGEMM-Ascend'","return (x['id']=='7c7db643272857967545' and x.get('revision')=='conv2d_pilots_rollout_v33' and '169112' in str(x.get('error'))) or (x['key'][0]=='DeepGEMM-Ascend'",1).replace("t['id'] in ('be0","t['id'] in ('7c7db643272857967545','be0",1).replace('python_v33.sh','python_v34.sh').replace("'revision':'conv2d_pilots_rollout_v33'","'revision':'gdn_fp16_gate_abi_v34'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
row={'at':time.time(),'revision':'gdn_fp16_gate_abi_v34','id':'7c7db643272857967545','evidence':'Original kernelconfig all4 dtype variants require beta/raw_g FP32; case gate/beta FP16. Original C++ API lines669-676 leaves A5 lowprecision inputs unchanged unless one isFP32; full checker169112 OpDef mismatch. integration700_fla_native.py already uses losslessFP32gate/beta.', 'change':'Top-level wrapper per-call g.float()/beta.float(); inputvalues exact,conversiontimed;kernel/compiler/checker unchanged'}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''\nfrom pathlib import Path
import base64,json,time
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v34')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for GDN FP16 gate ABI v34','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v34')
''';subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
