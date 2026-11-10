import ast,json,pathlib,subprocess,sys
sys.dont_write_bytecode=True
O=pathlib.Path(__file__).resolve().parent;B=O.parent/'fp16_grouped_registration2';N=O.parent/'bf16_reference_budget2'
for name in ('run_environment.sh','run_case.sh'):subprocess.run(['bash','-n',str(O/name)],check=True)
for name in ('run_environment.sh','run_case.sh'):
 expected=(B/name).read_text().replace('FP16','BF16').replace('fp16_grouped_registration2','bf16_grouped_registration1').replace('fp16_grouped_budget900_2','bf16_reference_budget2').replace('moe-dgrad-float16|moe-fwd-float16','moe-fwd-bfloat16')
 assert (O/name).read_text()==expected
constants={}
for node in ast.parse((O/'integration700_bf16_grouped_registration1.py').read_bytes()).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  try:constants[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
assert isinstance(constants['ROOT'],str)
historical=json.loads((N/'moe-fwd-bfloat16/result/RUNTIME.json').read_text())['environment']
assert all(constants['ENV'][k]==v for k,v in historical.items() if k in constants['ENV'])
pins=json.loads((O/'FROZEN_PINS.json').read_text())
site=pathlib.Path('/root/dyq/workspace/.venv/lib/python3.12/site-packages')
assert {str(p.resolve()) for p in site.glob('*.pth')}=={p for p in pins if p.endswith('.pth')}
assert str(site/'fla_npu_opp_env.py') in pins
recipe=(O/'run_case.sh').read_text()
assert '--timeout 900' in recipe and '--wait 600' in recipe and 'moe-fwd-bfloat16)' in recipe
assert 'moe-dgrad' not in recipe and 'Bmm' not in recipe
assert 'run_environment.sh' in recipe and '.venv/bin/python -B' in recipe
result=dict(passed=True,checks=['both shell recipes syntax checked without execution','exact published template environment/recipe retained except BF16 single-key names and historical protection root','AST constant extraction skips nonliteral ROOT=Path(ROOT)','environment constants agree with original successful BF16 RUNTIME','complete .pth set and fla bootstrap source are pinned','single Grouped NT key; explicit owner; wait 600 and timeout 900; original SDK9.2/venv/vendor'],recipe_executed=False,npu_runs=0,device_imports=0)
(O/'RECIPE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
