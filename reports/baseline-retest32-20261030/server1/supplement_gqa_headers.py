from pathlib import Path
import hashlib,json,base64,subprocess
h=Path(__file__).resolve().parent
src=Path('ascend_baselines/ops-transformer')
local=Path('.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer/op_impl/ai_core/tbe/classicpilot_transformer_impl/ascendc')
rows=[]
for p in (src/'attention/common/op_kernel').rglob('*'):
 if p.is_file():rows.append((p,Path('common/op_kernel')/p.relative_to(src/'attention/common/op_kernel')))
rows.append((src/'posembedding/apply_rotary_pos_emb/op_kernel/arch35/apply_rotary_pos_emb_common.h',Path('rotary_position_embedding/apply_rotary_pos_emb/op_kernel/arch35/apply_rotary_pos_emb_common.h')))
data=[]
for p,r in rows:
 b=p.read_bytes();assert b==(local/r).read_bytes(),str(r)
 data.append({'path':str(r),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
code='rows='+repr(data)+'''\nfrom pathlib import Path
import hashlib,base64,json,time
root=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1/deployments/ops_gqa/vendors/gqapairs_transformer/op_impl/ai_core/tbe/gqapairs_transformer_impl/ascendc')
assert root.is_dir()
for row in rows:
 p=root/row['path'];b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 assert p.read_bytes()==b
print(json.dumps({'at':time.time(),'verified_files':len(rows),'root':str(root)}))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True)
x={'deployment':json.loads(r.stdout),'files':[{k:v for k,v in d.items() if k!='data'} for d in data]};(h/'GQA_HEADER_CLOSURE.json').write_text(json.dumps(x,indent=2));print(x['deployment'])
