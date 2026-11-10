from pathlib import Path
import json,shutil
W=Path(__file__).resolve().parent;R=W.parents[2];O=R/'reports/baseline-integration-20261026/transformer_gdn';P=W/'recurrent_package';P.mkdir(exist_ok=True)
for f in (O/'recurrent_package').iterdir():
 if f.is_file():shutil.copyfile(f,P/f.name)
r=json.loads((P/'RUNTIME_RECIPE.json').read_text());r['required_environment']['GDN_CASE_OWNER']='baseline700-20261026/transformer_linear';r['guard']['owner']='baseline700-20261026/transformer_linear';(P/'RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2))
i='q35-gva-decode-b512-bfloat16';(W/'RECURRENT_CASES.json').write_text(json.dumps([i]));d=W/i;d.mkdir(exist_ok=True)
e=json.loads((W/'CASE_RESOURCE_ESTIMATES.json').read_text());e[i]=json.loads((O/'CASE_RESOURCE_ESTIMATES.json').read_text())[i];t=W/'EST.tmp';t.write_text(json.dumps(e,indent=2));t.replace(W/'CASE_RESOURCE_ESTIMATES.json')
s=(O/i/'child.sh').read_text().replace(str(O),str(W)).replace('selected-integration-20261026/transformer_gdn','baseline700-20261026/transformer_linear')
for line in s.splitlines():
 if line.startswith('export ') and '="'+str(d)+'/' in line:Path(line.split('="')[1].rstrip('"')).mkdir(exist_ok=True)
(d/'child.sh').write_text(s)
print('recurrent ready, capacity must be checked before guard')
