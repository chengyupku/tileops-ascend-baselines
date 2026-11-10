from pathlib import Path
import base64,subprocess,json,time,hashlib
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';files={};s=(r/'tools/baseline_retest32_server1_entry_v34.py').read_text();s=s.replace("if task['key'][0]=='ops-transformer' and task['key'][1] in", "if task['key'][:2]==['ops-transformer','AddFwdOp']:\n    import baseline_retest32_server1_transformer_add_v35\nelif task['key'][0]=='ops-transformer' and task['key'][1] in",1);files['tools/baseline_retest32_server1_entry_v35.py']=s
files['tools/baseline_retest32_server1_python_v35.sh']=(r/'tools/baseline_retest32_server1_python_v34.sh').read_text().replace('[ "$case_op" = Conv2dFwdOp ]; then','[ "$case_op" = Conv2dFwdOp ] || [ "$case_op" = AddFwdOp ]; then').replace('entry_v34','entry_v35')
files['tools/baseline_retest32_server1_transformer_add_v35.py']=(r/'tools/baseline_retest32_server1_transformer_add_v35.py').read_text()
p=r/'tools/run_baseline_retest32_server1.py';s=p.read_text();(h/'runner.before_v35.py').write_text(s);s=s.replace("t['id']!='7c7db643272857967545',","t['id'] not in ('1897b04e25e8bb1ad437','064a7141acb78172e294'),t['id']!='7c7db643272857967545',",1);s=s.replace("or (t['key'][1]=='Conv2dFwdOp'", "or (t['key'][1]=='AddFwdOp' and t['case']['params']['input'][1] in ('float32','bfloat16') and (t['id'] in ('1897b04e25e8bb1ad437','064a7141acb78172e294') or all(latest.get(i,{}).get('status')=='cached' for i in ('1897b04e25e8bb1ad437','064a7141acb78172e294')))) or (t['key'][1]=='Conv2dFwdOp'",1);s=s.replace("if len(chosen)==4 or", "if len(chosen)==4 or t['id'] in ('1897b04e25e8bb1ad437','064a7141acb78172e294') or",1).replace('python_v34.sh','python_v35.sh').replace("'revision':'gdn_fp16_gate_abi_v34'","'revision':'add_routing_acl_v35'");files['tools/run_baseline_retest32_server1.py']=s
for p,s in files.items():
 if p.endswith('.py'):compile(s,p,'exec')
 (r/p).write_text(s)
sources=['TileOPs/benchmarks/ascend_all_pairs/ops_transformer/routing_arithmetic.py','ascend_baselines/ops-transformer/moe/moe_finalize_routing_v2/op_host/op_api/aclnn_moe_finalize_routing_v2.h'];hashes={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in sources}
row={'at':time.time(),'revision':'add_routing_acl_v35','pilots':['1897b04e25e8bb1ad437','064a7141acb78172e294'],'new_condition':'Original public ACLNN symbol present in deployedARMvendor;bind exact7tensor,int64,output signature;original routing dispatch unchanged;FP32/BF16 fullpilots gate rollout','source_hashes':hashes}
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
code='files='+repr({p:base64.b64encode(s.encode()).decode() for p,s in files.items()})+'\nrow='+repr(row)+'''\nfrom pathlib import Path
import base64,json,time,hashlib
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1'
for p,digest in row['source_hashes'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==digest,p
for p,s in files.items():
 dest=r/p
 if dest.exists():(h/(dest.name+'.before_v35')).write_bytes(dest.read_bytes())
 tmp=dest.with_suffix('.deploy');tmp.write_bytes(base64.b64decode(s));tmp.replace(dest)
(h/'STOP').write_text(json.dumps({'reason':'drain for Add original ACLNN pilots v35','at':time.time()}))
with (h/'RETRIES.jsonl').open('a') as f:f.write(json.dumps(row)+'\\n')
print('deployed v35')
''';subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,check=True)
