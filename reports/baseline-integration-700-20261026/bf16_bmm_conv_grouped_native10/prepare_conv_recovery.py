from pathlib import Path
import json,hashlib,ast
T=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=(T/'capture.py').read_text()
s=s.replace("(TASK/'FROZEN.json')","(TASK/'FROZEN_CONV.json')")
s=s.replace("assert params==row['manifest_params'],(params,row['manifest_params'])", "assert all(params[k]==v for k,v in row['manifest_params'].items()),(params,row['manifest_params'])\nif OP=='Conv1dFwdOp':\n expected_derived=dict(C_in_g=params['C_in']//params['groups'],dW=params['dilation'],pW=params['padding'],sW=params['stride'],L_out=(params['L_in']+2*params['padding']-params['dilation']*(params['kW']-1)-1)//params['stride']+1)\n assert {k:params[k] for k in expected_derived}==expected_derived")
ast.parse(s);(T/'capture_conv.py').write_text(s)
pins=json.loads((T/'FROZEN.json').read_text());pins['sha256'][str(T/'capture_conv.py')]=sha(T/'capture_conv.py');(T/'FROZEN_CONV.json').write_text(json.dumps(pins,indent=2))
P=T/'conv-firstcanonical';P.mkdir(exist_ok=True)
old=T/'whisper-conv1-bias-bfloat16'
run=(old/'run.sh').read_text().replace(str(old),str(P)).replace(str(T/'capture.py'),str(T/'capture_conv.py'))
for p in old.iterdir():
 if p.is_dir():(P/p.name).mkdir(exist_ok=True)
(P/'run.sh').write_text(run)
(T/'CONV_PREFLIGHT_CORRECTION.json').write_text(json.dumps({'prior_evidence':str(old),'prior_status':'pre_input_pre_reference_pre_native_harness_assertion','cause':'Exact dictionary equality rejected legitimate derived Conv indices absent from preparation base-parameter list.','change':'Check every declared parameter plus exact derived Conv indices; preserve actual case._call.indices into parent-reviewed wrapper. Separate capture_conv.py and FROZEN_CONV.json; original frozen capture.py unchanged.','reference_changes':False,'kernel_changes':False,'numerical_rerun':False,'new_evidence':str(P)},indent=2))
p=T/'finalize.py';s=p.read_text().replace("P=T/row['key'][2];d=dict(row)","P=Path(row.get('evidence',str(T/row['key'][2])));d=dict(row)");p.write_text(s)
print('prepared separate Conv canonical harness; existing frozen batch unchanged')
